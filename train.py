import importlib
import sys

# --- ★★★ キャッシュクリアのためのコード ★★★ ---
if 'utils' in sys.modules:
    importlib.reload(sys.modules['utils'])
if 'model' in sys.modules:
    importlib.reload(sys.modules['model'])
if 'grid' in sys.modules:
    importlib.reload(sys.modules['grid'])
# --- ここまで ---

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pickle

from model import SocialModel
from utils import TrajectoryDataset
# --- ★★★ 高速化対応 ★★★ ---
# 新しいベクトル化された関数をインポートします
from grid import get_grid_mask_vectorized as getSequenceGridMask

def bivariate_loss(V_pred, V_trgt):
    """二変量正規分布の負の対数尤度損失を計算する"""
    muX = V_pred[:, :, 0]
    muY = V_pred[:, :, 1]
    sigX = torch.exp(V_pred[:, :, 2])
    sigY = torch.exp(V_pred[:, :, 3])
    rho = torch.tanh(V_pred[:, :, 4])
    
    x = V_trgt[:, :, 0]
    y = V_trgt[:, :, 1]

    zx = (x - muX) / sigX
    zy = (y - muY) / sigY
    
    rho = torch.clamp(rho, -0.9999, 0.9999)
    
    exponent = - (zx**2 - 2*rho*zx*zy + zy**2) / (2 * (1 - rho**2))
    log_pi_sig_sqrt = torch.log(2 * torch.pi * sigX * sigY * torch.sqrt(1 - rho**2))
    
    loss = (log_pi_sig_sqrt + exponent).sum()
    return loss

def vald(model, loader, args, device):
    """検証ループ関数"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, V_obs, A_obs, V_tr, A_tr) = batch

            obs_traj = obs_traj.squeeze(0).to(device)
            pred_traj_gt = pred_traj_gt.squeeze(0).to(device)
            obs_traj_rel = obs_traj_rel.squeeze(0).to(device)
            pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).to(device)
            
            num_peds = obs_traj.size(1)

            full_traj_rel_input = torch.cat((obs_traj_rel, pred_traj_gt_rel[:-1]), dim=0)
            full_traj_abs_input = torch.cat((obs_traj, pred_traj_gt[:-1]), dim=0)
            
            # --- ★★★ 高速化対応 ★★★ ---
            # ループを削除し、一度の呼び出しで全グリッドを計算
            grids = getSequenceGridMask(full_traj_abs_input, args.neighborhood_size, args.grid_size, args.use_cuda)

            hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
            cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
            
            peds_list = [torch.arange(num_peds).to(device) for _ in range(args.obs_len + args.pred_len)]
            lookup = {i: i for i in range(num_peds)}
            
            full_outputs, _, _ = model(full_traj_rel_input, grids, hidden_states, cell_states, peds_list, lookup)
            
            pred_outputs = full_outputs[args.obs_len-1:]

            loss = bivariate_loss(pred_outputs, pred_traj_gt_rel)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accumulation_steps', type=int, default=32, help='(Optional) Number of steps to accumulate gradients.')
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5) 
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--dataset', default='all', help='eth,hotel,univ,zara1,zara2,all')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size is 1 for this model')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--clip_grad', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--tag', default='social_lstm_model', help='tag for the model')
    
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.seq_length = args.obs_len + args.pred_len

    device = torch.device("cuda" if args.use_cuda else "cpu")
    print(f"Using device: {device}")

    all_dataset_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    datasets_to_load = all_dataset_names if args.dataset == 'all' else [args.dataset]
    
    train_datasets = []
    val_datasets = []
    print("Loading datasets...")
    for name in datasets_to_load:
        train_path = f'./datasets/{name}/train/'
        val_path = f'./datasets/{name}/val/'
        
        if os.path.exists(train_path):
            try:
                dset = TrajectoryDataset(train_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1, delim='\t')
                if len(dset) > 0:
                    train_datasets.append((name, dset))
                    print(f"  - Successfully loaded {name} training data ({len(dset)} sequences)")
            except (IndexError, ValueError) as e:
                print(f"  - Error loading {name} train data: {e}. Skipping.")

        if os.path.exists(val_path):
            try:
                dset = TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1, delim='\t')
                if len(dset) > 0:
                    val_datasets.append(dset)
                    print(f"  - Successfully loaded {name} validation data ({len(dset)} sequences)")
            except (IndexError, ValueError) as e:
                print(f"  - Error loading {name} val data: {e}. Skipping.")


    if not train_datasets:
        print("No valid training data found. Exiting.")
        return

    dset_val = ConcatDataset(val_datasets)
    loader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False)
    
    total_train_samples = sum(len(d[1]) for d in train_datasets)
    print(f"Total training sequences: {total_train_samples}")
    print(f"Total validation sequences: {len(dset_val)}")

    model = SocialModel(args).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    checkpoint_dir = f'./checkpoint/{args.tag}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_val_loss = float('inf')
    print("******************************")
    print("Training initiating....")
    for epoch in range(1, args.num_epochs + 1):
        epoch_total_loss = 0
        model.train()
        print(f"--- Epoch {epoch}/{args.num_epochs} Train ---")
        
        optimizer.zero_grad()
        
        for dset_name, dset_train in train_datasets:
            loader_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
            print(f"  Training on: {dset_name}")
            
            for batch_idx, batch in enumerate(loader_train):
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
                 loss_mask, V_obs, A_obs, V_tr, A_tr) = batch

                obs_traj = obs_traj.squeeze(0).to(device)
                pred_traj_gt = pred_traj_gt.squeeze(0).to(device)
                obs_traj_rel = obs_traj_rel.squeeze(0).to(device)
                pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).to(device)
                
                num_peds = obs_traj.size(1)

                full_traj_rel_input = torch.cat((obs_traj_rel, pred_traj_gt_rel[:-1]), dim=0)
                full_traj_abs_input = torch.cat((obs_traj, pred_traj_gt[:-1]), dim=0)
                
                # --- ★★★ 高速化対応 ★★★ ---
                # ループを削除し、一度の呼び出しで全グリッドを計算
                grids = getSequenceGridMask(full_traj_abs_input, args.neighborhood_size, args.grid_size, args.use_cuda)

                hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
                cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
                
                peds_list = [torch.arange(num_peds).to(device) for _ in range(args.obs_len + args.pred_len)]
                lookup = {i: i for i in range(num_peds)}

                full_outputs, _, _ = model(full_traj_rel_input, grids, hidden_states, cell_states, peds_list, lookup)
                pred_outputs = full_outputs[args.obs_len-1:]
                
                loss = bivariate_loss(pred_outputs, pred_traj_gt_rel) / args.accumulation_steps
                
                epoch_total_loss += loss.item() * args.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    if args.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()

        avg_train_loss = epoch_total_loss / total_train_samples if total_train_samples > 0 else 0
        
        val_loss = vald(model, loader_val, args, device)
        
        print(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'val_best.pth'))
            print(f"  *** New best model saved at epoch {epoch} ***")
        
        print("-" * 30)

if __name__ == '__main__':
    main()
