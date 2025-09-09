# train.py (Social-STGCNNの構造をSocial-LSTMに適用した最終版)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import pickle
from utils import DataLoader
from model import SocialModel
from grid import get_grid_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rel_to_abs(rel_traj, start_pos):
    """相対座標を絶対座標に変換するヘルパー関数"""
    # rel_traj: (batch, pred_len, num_peds, 2), start_pos: (batch, num_peds, 2)
    pred_traj_abs = torch.zeros_like(rel_traj)
    current_pos = start_pos.clone()
    for t in range(rel_traj.shape[1]):
        current_pos = current_pos + rel_traj[:, t, :, :]
        pred_traj_abs[:, t, :, :] = current_pos
    return pred_traj_abs

def train_epoch(epoch, model, train_loaders, optimizer, args):
    """1エポック分の学習を行う関数"""
    model.train()
    epoch_loss = 0
    num_batches_total = sum(loader.get_num_batches() for loader in train_loaders)
    
    for loader in train_loaders:
        loader.reset_batch_pointer()
        for _ in range(loader.get_num_batches()):
            obs_rel, obs_abs, pred_gt_abs = loader.next_batch()
            obs_rel, obs_abs = obs_rel.to(device), obs_abs.to(device)
            
            # 正解の相対座標を計算
            pred_gt_rel = torch.zeros_like(pred_gt_abs)
            pred_gt_rel[:, 0, :, :] = pred_gt_abs[:, 0, :, :] - obs_abs[:, -1, :, :]
            pred_gt_rel[:, 1:, :, :] = pred_gt_abs[:, 1:, :, :] - pred_gt_abs[:, :-1, :, :]
            pred_gt_rel = pred_gt_rel.to(device)

            # グリッドマスクを計算
            grids = []
            for t in range(args.obs_len):
                frame_data = obs_abs[:, t]
                batch_grids = [get_grid_mask(frame_data[b], args.neighborhood_size, args.grid_size) for b in range(args.batch_size)]
                grids.append(torch.stack(batch_grids, dim=0).unsqueeze(1))
            grids = torch.cat(grids, dim=1)

            optimizer.zero_grad()
            pred_fake_rel = model(obs_rel, grids)
            
            # NaN（パディングされた歩行者）を除外してロスを計算
            loss = nn.MSELoss()(pred_fake_rel[~torch.isnan(pred_gt_rel)], pred_gt_rel[~torch.isnan(pred_gt_rel)])
            if torch.isnan(loss): continue

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / num_batches_total if num_batches_total > 0 else 0
    print(f"TRAIN:\t Epoch: {epoch}\t Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def test_epoch(epoch, model, test_loader, args):
    """1エポック分の評価（ADE/FDE計算）を行う関数"""
    model.eval()
    total_ade, total_fde = 0, 0
    
    with torch.no_grad():
        test_loader.reset_batch_pointer()
        for _ in range(test_loader.get_num_batches()):
            obs_rel, obs_abs, pred_gt_abs = test_loader.next_batch()
            obs_rel, obs_abs, pred_gt_abs = obs_rel.to(device), obs_abs.to(device), pred_gt_abs.to(device)

            grids = []
            for t in range(args.obs_len):
                frame_data = obs_abs[:, t]
                batch_grids = [get_grid_mask(frame_data[b], args.neighborhood_size, args.grid_size) for b in range(args.batch_size)]
                grids.append(torch.stack(batch_grids, dim=0).unsqueeze(1))
            grids = torch.cat(grids, dim=1)
            
            pred_fake_rel = model(obs_rel, grids)
            pred_fake_abs = rel_to_abs(pred_fake_rel, obs_abs[:, -1])
            
            # NaN（パディング）を除外してADE/FDEを計算
            mask = ~torch.isnan(pred_gt_abs[:, :, :, 0])
            if torch.sum(mask) == 0: continue
            
            ade = torch.mean(torch.norm(pred_gt_abs[mask] - pred_fake_abs[mask], dim=-1))
            
            final_frame_mask = ~torch.isnan(pred_gt_abs[:, -1, :, 0])
            if torch.sum(final_frame_mask) == 0: continue
            
            fde = torch.mean(torch.norm(pred_gt_abs[:, -1][final_frame_mask] - pred_fake_abs[:, -1][final_frame_mask], dim=-1))
            
            total_ade += ade.item()
            total_fde += fde.item()

    avg_ade = total_ade / test_loader.get_num_batches()
    avg_fde = total_fde / test_loader.get_num_batches()

    print(f"VALD:\t Epoch: {epoch}\t ADE: {avg_ade:.4f}\t FDE: {avg_fde:.4f}")
    return avg_ade, avg_fde

def main():
    parser = argparse.ArgumentParser()
    # Data specific parameters
    parser.add_argument('--data_path', default='./datasets', help='Path to dataset directory')
    parser.add_argument('--dataset', default='eth', help='Dataset to test on (eth, hotel, zara01, zara02, univ)')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--force_preprocess', action='store_true', default=False, help='Force pre-processing of data')

    # Training specific parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tag', default='social_lstm_tag', help='Personal tag for the model')

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0, help='Neighborhood size for social grid')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size for social grid')
    
    args = parser.parse_args()

    print('*' * 30)
    print("Training initiating....")
    print(args)

    # Data prep
    all_datasets = ['eth', 'hotel', 'zara01', 'zara02', 'univ']
    if args.dataset not in all_datasets:
        raise ValueError("Error: --dataset must be one of " + ", ".join(all_datasets))
    
    train_names = [name for name in all_datasets if name != args.dataset]
    test_name = args.dataset
    
    print(f"--- Training on: {', '.join(train_names)} ---")
    print(f"--- Testing on: {test_name} ---")
    
    train_loaders = [DataLoader(os.path.join(args.data_path, name), args.batch_size, args.obs_len, args.pred_len, forcePreProcess=args.force_preprocess) for name in train_names]
    test_loader = DataLoader(os.path.join(args.data_path, test_name), args.batch_size, args.obs_len, args.pred_len, forcePreProcess=args.force_preprocess)

    # Defining the model
    model = SocialModel(args).to(device)

    # Training settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_dir = './checkpoint/' + args.tag + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # Training
    metrics = {'train_loss': [], 'val_ade': [], 'val_fde': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_ade': float('inf')}

    print('Training started ...')
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(epoch, model, train_loaders, optimizer, args)
        val_ade, val_fde = test_epoch(epoch, model, test_loader, args)
        
        metrics['train_loss'].append(train_loss)
        metrics['val_ade'].append(val_ade)
        metrics['val_fde'].append(val_fde)

        if val_ade < constant_metrics['min_val_ade']:
            constant_metrics['min_val_ade'] = val_ade
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')

        print('*' * 30)
        print(f"Epoch: {args.tag} : {epoch}")
        print(f"Current Train Loss: {metrics['train_loss'][-1]:.4f}")
        print(f"Current Val ADE/FDE: {metrics['val_ade'][-1]:.4f} / {metrics['val_fde'][-1]:.4f}")
        print(f"Best Val ADE: {constant_metrics['min_val_ade']:.4f} at Epoch {constant_metrics['min_val_epoch']}")
        print('*' * 30)
        
        with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)
        
        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)

if __name__ == '__main__':
    main()
