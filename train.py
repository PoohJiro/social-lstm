# train.py (全trainデータで学習し、指定したvalデータで評価する最終版)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import argparse
import os
from utils import TrajectoryDataset, seq_collate
from model import SocialModel
from grid import get_grid_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rel_to_abs(rel_traj, start_pos):
    """相対座標を絶対座標に変換"""
    abs_traj = torch.zeros_like(rel_traj)
    current_pos = start_pos.clone()
    for t in range(rel_traj.shape[1]):
        current_pos = current_pos + rel_traj[:, t, :, :]
        abs_traj[:, t, :, :] = current_pos
    return abs_traj

def train_epoch(epoch, model, loader_train, optimizer, args):
    """1エポック分の学習"""
    model.train()
    epoch_loss = 0
    for batch in loader_train:
        # DataLoaderの出力: obs_abs, pred_abs, obs_rel
        obs_abs, pred_abs, obs_rel = [t.to(device) for t in batch]
        
        # 正解の相対座標を計算
        pred_gt_rel = torch.zeros_like(pred_abs)
        pred_gt_rel[:, 0, :, :] = pred_abs[:, 0, :, :] - obs_abs[:, -1, :, :]
        pred_gt_rel[:, 1:, :, :] = pred_abs[:, 1:, :, :] - pred_abs[:, :-1, :, :]
        
        grids = []
        for t in range(args.obs_len):
            frame_data = obs_abs[:, t]
            batch_grids = [get_grid_mask(frame, args.neighborhood_size, args.grid_size) for frame in frame_data]
            grids.append(torch.stack(batch_grids, dim=0).unsqueeze(1))
        grids = torch.cat(grids, dim=1)

        optimizer.zero_grad()
        pred_fake_rel = model(obs_rel, grids)
        
        loss = nn.MSELoss()(pred_fake_rel[~torch.isnan(pred_gt_rel)], pred_gt_rel[~torch.isnan(pred_gt_rel)])
        if torch.isnan(loss): continue
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader_train)
    print(f"TRAIN:\t Epoch: {epoch}\t Loss: {avg_loss:.4f}")

def test_epoch(epoch, model, loader_val, args):
    """1エポック分の評価"""
    model.eval()
    total_ade, total_fde = 0, 0
    with torch.no_grad():
        for batch in loader_val:
            obs_abs, pred_gt_abs, obs_rel = [t.to(device) for t in batch]
            
            grids = []
            for t in range(args.obs_len):
                frame_data = obs_abs[:, t]
                batch_grids = [get_grid_mask(frame, args.neighborhood_size, args.grid_size) for frame in frame_data]
                grids.append(torch.stack(batch_grids, dim=0).unsqueeze(1))
            grids = torch.cat(grids, dim=1)
            
            pred_fake_rel = model(obs_rel, grids)
            pred_fake_abs = rel_to_abs(pred_fake_rel, obs_abs[:, -1])
            
            mask = ~torch.isnan(pred_gt_abs[:, :, :, 0])
            if not mask.any(): continue
            ade = torch.mean(torch.norm(pred_gt_abs[mask] - pred_fake_abs[mask], dim=-1))
            
            final_mask = ~torch.isnan(pred_gt_abs[:, -1, :, 0])
            if not final_mask.any(): continue
            fde = torch.mean(torch.norm(pred_gt_abs[:, -1][final_mask] - pred_fake_abs[:, -1][final_mask], dim=-1))

            total_ade += ade.item()
            total_fde += fde.item()

    avg_ade = total_ade / len(loader_val)
    avg_fde = total_fde / len(loader_val)
    print(f"VALD:\t Epoch: {epoch}\t ADE: {avg_ade:.4f}\t FDE: {avg_fde:.4f}")
    return avg_ade

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./datasets', help='Path to dataset directory')
    parser.add_argument('--dataset_val', default='eth', help='Dataset to use for validation (eth, hotel, zara01, zara02, univ)')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0)
    parser.add_argument('--grid_size', type=int, default=4)
    args = parser.parse_args()

    # --- Data prep ---
    all_datasets = ['eth', 'hotel', 'zara01', 'zara02', 'univ']
    
    # 全ての'train'フォルダのパスリストを作成
    train_paths = [os.path.join(args.data_path, name, 'train') for name in all_datasets]
    # 各パスからTrajectoryDatasetオブジェクトを作成
    train_datasets = [TrajectoryDataset(path, obs_len=args.obs_len, pred_len=args.pred_len) for path in train_paths]
    # 全ての学習データセットを一つに結合
    concat_train_dataset = ConcatDataset(train_datasets)
    
    # 結合したデータセットから学習用DataLoaderを作成
    loader_train = DataLoader(concat_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=seq_collate)
    
    # --dataset_valで指定された'val'フォルダから評価用DataLoaderを作成
    val_path = os.path.join(args.data_path, args.dataset_val, 'val')
    dset_val = TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len)
    loader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=seq_collate)

    print(f"--- Training on all 'train' folders ---")
    print(f"--- Validating on: {args.dataset_val}/val ---")

    # --- Model, Optimizer, and Training Loop ---
    model = SocialModel(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_ade = float('inf')
    for epoch in range(args.num_epochs):
        train_epoch(epoch, model, loader_train, optimizer, args)
        val_ade = test_epoch(epoch, model, loader_val, args)

        if val_ade < best_ade:
            best_ade = val_ade
            print(f"Saving best model with ADE: {best_ade:.4f}")
            torch.save(model.state_dict(), f'social_lstm_all_train_{args.dataset_val}_val_best.pth')

if __name__ == '__main__':
    main()
