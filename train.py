import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import time
import pickle

from model import SocialModel # NOTE: This model.py also needs to be adapted
from utils import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    # --- 1. Load Data ---
    # Create separate DataLoaders for train, val, test
    # Assumes a directory structure like: ./datasets/eth/, ./datasets/hotel/, etc.
    datasets = ['eth', 'hotel', 'zara01', 'zara02', 'univ']
    
    # For this example, we'll train on one dataset. You can loop through them for a full experiment.
    # The Social-STGCNN "leave-one-out" approach is more complex. We'll start with a simple train/test split.
    dataset_name = 'eth' # Example dataset
    data_dir = os.path.join('./datasets', dataset_name)
    
    # For simplicity, we'll use one dataset for train/val and another for test.
    # A more robust approach would be to use the standard train/val/test splits.
    print(f"--- Loading Training Data: {dataset_name} ---")
    train_loader = DataLoader(data_dir, batch_size=args.batch_size, seq_length=args.obs_len, pred_length=args.pred_len)
    
    # Example of using a different dataset for validation
    val_dataset_name = 'hotel'
    val_data_dir = os.path.join('./datasets', val_dataset_name)
    print(f"--- Loading Validation Data: {val_dataset_name} ---")
    val_loader = DataLoader(val_data_dir, batch_size=args.batch_size, seq_length=args.obs_len, pred_length=args.pred_len)
    
    # --- 2. Initialize Model, Optimizer, Loss ---
    net = SocialModel(args).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # Use Mean Squared Error for simplicity

    # --- 3. Training Loop ---
    print("--- Starting Training ---")
    best_ade = float('inf')
    for epoch in range(args.num_epochs):
        net.train()
        epoch_loss = 0
        
        for batch_idx in range(train_loader.get_num_batches()):
            obs_traj, pred_traj_gt = train_loader.next_batch()
            obs_traj, pred_traj_gt = obs_traj.to(device), pred_traj_gt.to(device)

            optimizer.zero_grad()
            
            # The model's forward pass needs to be adapted to this input
            pred_traj_fake = net(obs_traj)
            
            loss = criterion(pred_traj_fake, pred_traj_gt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        epoch_loss /= train_loader.get_num_batches()
        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {epoch_loss:.4f}")

        # --- 4. Validation Loop ---
        net.eval()
        total_ade = 0
        total_fde = 0
        
        with torch.no_grad():
            for batch_idx in range(val_loader.get_num_batches()):
                obs_traj, pred_traj_gt = val_loader.next_batch()
                obs_traj, pred_traj_gt = obs_traj.to(device), pred_traj_gt.to(device)

                pred_traj_fake = net(obs_traj)
                
                # Calculate ADE and FDE
                # L2 distance between predicted and ground truth
                ade = torch.mean(torch.norm(pred_traj_gt - pred_traj_fake, p=2, dim=2))
                # L2 distance at the final timestep
                fde = torch.mean(torch.norm(pred_traj_gt[:, -1, :] - pred_traj_fake[:, -1, :], p=2, dim=1))

                total_ade += ade.item()
                total_fde += fde.item()

        avg_ade = total_ade / val_loader.get_num_batches()
        avg_fde = total_fde / val_loader.get_num_batches()

        print(f"Validation ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")

        # Save the best model
        if avg_ade < best_ade:
            best_ade = avg_ade
            print("Saving best model...")
            torch.save(net.state_dict(), f'social_lstm_model_{dataset_name}_best.pth')

    print("--- Training Finished ---")

def test(args):
    # --- 5. Test Function ---
    print("--- Starting Testing ---")
    dataset_name = 'zara01' # Example test dataset
    test_data_dir = os.path.join('./datasets', dataset_name)
    print(f"--- Loading Test Data: {dataset_name} ---")
    test_loader = DataLoader(test_data_dir, batch_size=args.batch_size, seq_length=args.obs_len, pred_length=args.pred_len)

    net = SocialModel(args).to(device)
    net.load_state_dict(torch.load(f'social_lstm_model_eth_best.pth')) # Load the trained model
    net.eval()

    total_ade = 0
    total_fde = 0
    
    with torch.no_grad():
        for batch_idx in range(test_loader.get_num_batches()):
            obs_traj, pred_traj_gt = test_loader.next_batch()
            obs_traj, pred_traj_gt = obs_traj.to(device), pred_traj_gt.to(device)

            pred_traj_fake = net(obs_traj)
            
            ade = torch.mean(torch.norm(pred_traj_gt - pred_traj_fake, p=2, dim=2))
            fde = torch.mean(torch.norm(pred_traj_gt[:, -1, :] - pred_traj_fake[:, -1, :], p=2, dim=1))

            total_ade += ade.item()
            total_fde += fde.item()

    avg_ade = total_ade / test_loader.get_num_batches()
    avg_fde = total_fde / test_loader.get_num_batches()

    print("--- Test Results ---")
    print(f"Dataset: {dataset_name}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- Data parameters ---
    parser.add_argument('--obs_len', type=int, default=8, help='Observation sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length')

    # --- Training parameters ---
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    
    # --- Model parameters ---
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    parser.add_argument('--gru', action="store_true", default=False, help='True: GRU cell, False: LSTM cell')
    
    # --- Execution parameters ---
    parser.add_argument('--test_only', action='store_true', help='Run testing only')
    
    args = parser.parse_args()
    
    if not args.test_only:
        train(args)
    else:
        test(args)
