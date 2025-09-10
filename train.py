# train.py (全trainデータで学習し、指定したvalデータで検証する最終版)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import argparse
import os

# 以下のファイルが同じディレクトリにあることを確認してください
from utils import TrajectoryDataset, seq_collate
from model import SocialModel
from grid import get_grid_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rel_to_abs(rel_traj, start_pos):
    """
    相対座標の軌跡を絶対座標に変換するヘルパー関数
    Args:
        rel_traj (torch.Tensor): 相対座標の軌跡 (batch, pred_len, num_peds, 2)
        start_pos (torch.Tensor): 各軌跡の開始地点となる絶対座標 (batch, num_peds, 2)
    Returns:
        torch.Tensor: 絶対座標に変換された軌跡
    """
    # (batch, pred_len, num_peds, 2) -> (batch, num_peds, pred_len, 2)
    rel_traj = rel_traj.permute(0, 2, 1, 3)
    abs_traj = torch.zeros_like(rel_traj)
    current_pos = start_pos.clone()
    for t in range(rel_traj.shape[2]):
        current_pos = current_pos + rel_traj[:, :, t, :]
        abs_traj[:, :, t, :] = current_pos
    return abs_traj.permute(0, 2, 1, 3)

def train_epoch(model, loader_train, optimizer, args):
    """1エポック分の学習を行う"""
    model.train()
    epoch_loss = 0
    for batch in loader_train:
        # DataLoaderからの出力を受け取る
        obs_abs, pred_gt_abs, obs_rel = [t.to(device) for t in batch]
        
        # 正解データとなる相対座標を計算
        pred_gt_rel = torch.zeros_like(pred_gt_abs)
        pred_gt_rel[:, 0, :, :] = pred_gt_abs[:, 0, :, :] - obs_abs[:, -1, :, :]
        pred_gt_rel[:, 1:, :, :] = pred_gt_abs[:, 1:, :, :] - pred_gt_abs[:, :-1, :, :]

        # グリッドマスクを計算
        grids = []
        for t in range(args.obs_len):
            frame_data = obs_abs[:, t]
            batch_grids = [get_grid_mask(frame, args.neighborhood_size, args.grid_size) for frame in frame_data]
            grids.append(torch.stack(batch_grids, dim=0).unsqueeze(1))
        grids = torch.cat(grids, dim=1)

        optimizer.zero_grad()
        pred_fake_rel = model(obs_rel, grids)
        
        # NaN（パディングされた歩行者）を除外してロスを計算
        mask = ~torch.isnan(pred_gt_rel)
        if not mask.any(): continue # バッチ内に有効なデータがない場合はスキップ
        
        loss = nn.MSELoss()(pred_fake_rel[mask], pred_gt_rel[mask])
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader_train)

def val_epoch(model, loader_val, args):
    """1エポック分の検証 (ADE/FDE計算) を行う"""
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
            # 予測された相対座標を、観測の最後の絶対座標を起点に絶対座標へ変換
            pred_fake_abs = rel_to_abs(pred_fake_rel, obs_abs[:, -1])
            
            # NaNを除外してADE/FDEを計算
            mask = ~torch.isnan(pred_gt_abs[:, :, :, 0])
            if not mask.any(): continue
            
            ade = torch.mean(torch.norm(pred_gt_abs[mask] - pred_fake_abs[mask], p=2, dim=-1))
            
            final_mask = ~torch.isnan(pred_gt_abs[:, -1, :, 0])
            if not final_mask.any(): continue
            
            fde = torch.mean(torch.norm(pred_gt_abs[:, -1][final_mask] - pred_fake_abs[:, -1][final_mask], p=2, dim=-1))

            total_ade += ade.item()
            total_fde += fde.item()

    avg_ade = total_ade / len(loader_val)
    avg_fde = total_fde / len(loader_val)
    return avg_ade, avg_fde

def main():
    parser = argparse.ArgumentParser()
    # データ関連の引数
    parser.add_argument('--data_path', default='./datasets', help='Path to dataset directory')
    parser.add_argument('--dataset_val', default='eth', help='Dataset to use for validation (eth, hotel, zara1, zara2, univ)')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)

    # 学習関連の引数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)

    # モデル関連の引数
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0)
    parser.add_argument('--grid_size', type=int, default=4)
    args = parser.parse_args()

    # --- 1. データの準備 ---
    all_dataset_names = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
    
    # 全ての'train'フォルダのパスリストを作成
    train_paths = [os.path.join(args.data_path, name, 'train') for name in all_dataset_names]
    # 各パスからTrajectoryDatasetオブジェクトを作成
    train_datasets = [TrajectoryDataset(path, obs_len=args.obs_len, pred_len=args.pred_len) for path in train_paths]
    
    # 全ての学習データセットを一つに結合
    print(f"--- Combining {len(train_datasets)} training datasets ---")
    concat_train_dataset = ConcatDataset(train_datasets)
    loader_train = DataLoader(concat_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=seq_collate)
    
    # --dataset_valで指定された'val'フォルダから検証用DataLoaderを作成
    val_path = os.path.join(args.data_path, args.dataset_val, 'val')
    print(f"--- Validating on: {val_path} ---")
    dset_val = TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len)
    loader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=seq_collate)
    
    # --- 2. モデルとオプティマイザの準備 ---
    model = SocialModel(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- 3. 学習と検証のループ ---
    best_ade = float('inf')
    print("--- Starting Training ---")
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, loader_train, optimizer, args)
        val_ade, val_fde = val_epoch(model, loader_val, args)
        print(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {train_loss:.4f}, Val ADE: {val_ade:.4f}, Val FDE: {val_fde:.4f}")

        # 最良のADEを記録したモデルを保存
        if val_ade < best_ade:
            best_ade = val_ade
            print(f"****** Best Val ADE so far: {best_ade:.4f}. Saving model... ******")
            torch.save(model.state_dict(), f'social_lstm_val_on_{args.dataset_val}_best.pth')

if __name__ == '__main__':
    main()
