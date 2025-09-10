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
from grid import getSequenceGridMask

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

def train(epoch, model, loader, optimizer, args, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, V_obs, A_obs, V_tr, A_tr) = batch

        # ★★★ エラー修正箇所 ★★★
        # 不要なpermuteを削除し、utils.pyからのデータをそのまま使います
        obs_traj = obs_traj.squeeze(0).to(device)
        obs_traj_rel = obs_traj_rel.squeeze(0).to(device)
        pred_traj_gt = pred_traj_gt.squeeze(0).to(device)
        pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).to(device)
        
        num_peds = obs_traj.size(1)

        grids = []
        # ループはobs_trajの最初の次元（シーケンス長）を正しく反復します
        for t in range(args.obs_len):
            grid = getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size, args.use_cuda)
            grids.append(grid)

        hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
        cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
        
        peds_list = [torch.arange(num_peds).to(device) for _ in range(args.obs_len + args.pred_len)]
        lookup = {i: i for i in range(num_peds)}

        optimizer.zero_grad()
        
        outputs, _, _ = model(obs_traj_rel, grids, hidden_states, cell_states, peds_list, lookup)
        
        loss = bivariate_loss(outputs, pred_traj_gt_rel)
        total_loss += loss.item()
        
        loss.backward()
        
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    print(f'Epoch: {epoch}, Train Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
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
    print("Loading datasets...")
    for name in datasets_to_load:
        train_path = f'./datasets/{name}/train/'
        if os.path.exists(train_path):
            try:
                dset = TrajectoryDataset(train_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1, delim='\t')
                if len(dset) > 0:
                    train_datasets.append(dset)
                    print(f"  - Successfully loaded {name} training data ({len(dset)} sequences)")
                else:
                    print(f"  - Warning: Skipped {name} training data (no valid sequences found).")
            except (IndexError, ValueError) as e:
                print(f"  - Error: Failed to load {name} training data due to a data format issue. Skipping.")
                print(f"    (Details: {e})")

    if not train_datasets:
        print("No valid training data found. Exiting.")
        return

    dset_train = ConcatDataset(train_datasets)
    
    loader_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
    print(f"Total training sequences combined: {len(dset_train)}")

    model = SocialModel(args).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    checkpoint_dir = f'./checkpoint/{args.tag}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("******************************")
    print("Training initiating....")
    for epoch in range(1, args.num_epochs + 1):
        train(epoch, model, loader_train, optimizer, args, device)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_{epoch}.pth'))
            print(f"Model saved at epoch {epoch}")

if __name__ == '__main__':
    main()
