import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import time
from model import SocialModel # ★★★ この model.py も後で修正が必要です ★★★
from utils import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    """
    モデルの学習と検証を行う関数
    """
    all_datasets = ['eth', 'hotel', 'zara01', 'zara02', 'univ']
    
    # テスト用データセットを除外し、残りを学習用データセットとする
    train_dataset_names = [name for name in all_datasets if name != args.dataset]
    test_dataset_name = args.dataset

    print(f"--- Training on: {', '.join(train_dataset_names)} ---")
    print(f"--- Testing on: {test_dataset_name} ---")

    # --- 1. Load Data ---
    # 複数の学習データセットを結合して1つのDataLoaderを作成
    train_loaders = [DataLoader(os.path.join('./datasets', name), batch_size=args.batch_size, seq_length=args.obs_len, pred_length=args.pred_len) for name in train_dataset_names]
    # テスト用データセットのDataLoaderを作成
    test_loader = DataLoader(os.path.join('./datasets', test_dataset_name), batch_size=args.batch_size, seq_length=args.obs_len, pred_length=args.pred_len)

    # --- 2. Initialize Model, Optimizer, Loss ---
    net = SocialModel(args).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # 平均二乗誤差を損失関数として使用

    # --- 3. Training Loop ---
    print("--- Starting Training ---")
    best_ade = float('inf')
    for epoch in range(args.num_epochs):
        net.train()
        epoch_loss = 0
        
        # 各データセットローダーで学習
        for loader in train_loaders:
            loader.reset_batch_pointer()
            for _ in range(loader.get_num_batches()):
                obs_traj, pred_traj_gt = loader.next_batch()
                obs_traj, pred_traj_gt = obs_traj.to(device), pred_traj_gt.to(device)

                optimizer.zero_grad()
                
                # ★★★ model.py の forward がこの入力を受け取るように修正が必要 ★★★
                pred_traj_fake = net(obs_traj)
                
                loss = criterion(pred_traj_fake, pred_traj_gt)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

        epoch_loss /= sum(loader.get_num_batches() for loader in train_loaders)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {epoch_loss:.4f}")

        # --- 4. Validation (Test) Loop ---
        current_ade, current_fde = test(args, net, test_loader)

        # 最良モデルの保存
        if current_ade < best_ade:
            best_ade = current_ade
            print("Saving best model...")
            torch.save(net.state_dict(), f'social_lstm_model_{test_dataset_name}_best.pth')

    print("--- Training Finished ---")


def test(args, net, test_loader):
    """
    モデルのテストを行い、ADEとFDEを計算する関数
    """
    net.eval()
    total_ade, total_fde = 0, 0
    
    with torch.no_grad():
        test_loader.reset_batch_pointer()
        for _ in range(test_loader.get_num_batches()):
            obs_traj, pred_traj_gt = test_loader.next_batch()
            obs_traj, pred_traj_gt = obs_traj.to(device), pred_traj_gt.to(device)

            pred_traj_fake = net(obs_traj)
            
            # ADE/FDEの計算
            # 予測全体と正解のL2距離
            ade = torch.mean(torch.norm(pred_traj_gt - pred_traj_fake, p=2, dim=2))
            # 最終地点と正解の最終地点のL2距離
            fde = torch.mean(torch.norm(pred_traj_gt[:, -1, :] - pred_traj_fake[:, -1, :], p=2, dim=1))

            total_ade += ade.item()
            total_fde += fde.item()

    avg_ade = total_ade / test_loader.get_num_batches()
    avg_fde = total_fde / test_loader.get_num_batches()

    print(f"Test Results -- ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")
    return avg_ade, avg_fde


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- Data parameters ---
    parser.add_argument('--dataset', default='eth', help='Dataset to be used for testing (eth, hotel, zara01, zara02, univ)')
    parser.add_argument('--obs_len', type=int, default=8, help='Observation sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length')

    # --- Training parameters ---
    parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # --- Model parameters ---
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128, help='Size of RNN hidden state')
    parser.add_argument('--gru', action="store_true", default=False, help='True: GRU cell, False: LSTM cell')
    
    args = parser.parse_args()
    
    train(args)
