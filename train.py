import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pickle

from model import SocialModel
# SocialModelと互換性のあるutils.pyを想定
from utils import TrajectoryDataset 

def bivariate_loss(V_pred, V_trgt):
    """二変量正規分布の負の対数尤度損失を計算する"""
    muX = V_pred[:, :, 0]
    muY = V_pred[:, :, 1]
    sigX = V_pred[:, :, 2]
    sigY = V_pred[:, :, 3]
    rho = V_pred[:, :, 4]
    
    x = V_trgt[:, :, 0]
    y = V_trgt[:, :, 1]

    # 正規化項
    norm = 1.0 / (2.0 * torch.pi * sigX * sigY * torch.sqrt(1.0 - rho**2))
    
    # 指数部分
    Z = ((x - muX) / sigX)**2 + ((y - muY) / sigY)**2 - (2.0 * rho * (x - muX) * (y - muY)) / (sigX * sigY)
    exponent = -Z / (2.0 * (1.0 - rho**2))
    
    # 損失計算
    loss = -torch.log(norm * torch.exp(exponent))
    return loss.sum()

def getSequenceGridMask(sequence, neighborhood_size, grid_size, use_cuda):
    """シーケンスのグリッドマスクを取得する"""
    num_peds = sequence.size(0)
    mask_shape = (num_peds, num_peds, grid_size * grid_size)
    grid_mask = torch.zeros(mask_shape)
    if use_cuda:
        grid_mask = grid_mask.cuda()

    for i in range(num_peds):
        for j in range(num_peds):
            if i == j:
                continue
            rel_pos = sequence[j] - sequence[i]
            
            # セルのインデックスを計算
            cell_x = int((rel_pos[0] + neighborhood_size) / (2 * neighborhood_size / grid_size))
            cell_y = int((rel_pos[1] + neighborhood_size) / (2 * neighborhood_size / grid_size))

            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                cell_idx = cell_y * grid_size + cell_x
                grid_mask[i, j, int(cell_idx)] = 1
                
    return grid_mask

def train(epoch, model, loader, optimizer, args, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        # 10個の要素をバッチから受け取る
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, V_obs, A_obs, V_tr, A_tr) = batch

        # SocialModelの入力形式に合わせる (batch, feat, seq) -> (seq, peds, feat)
        obs_traj = obs_traj.squeeze(0).permute(2, 0, 1).to(device)
        obs_traj_rel = obs_traj_rel.squeeze(0).permute(2, 0, 1).to(device)
        pred_traj_gt = pred_traj_gt.squeeze(0).permute(2, 0, 1).to(device)
        pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).permute(2, 0, 1).to(device)
        
        num_peds = obs_traj.size(1)

        # グリッドマスクを各タイムステップで作成
        grids = []
        for t in range(args.obs_len):
            grid = getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size, args.use_cuda)
            grids.append(grid)

        # 隠れ状態とセル状態を初期化
        hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
        cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
        
        peds_list = [torch.arange(num_peds).to(device) for _ in range(args.obs_len + args.pred_len)]
        lookup = {i: i for i in range(num_peds)}

        optimizer.zero_grad()
        
        # モデルに6つの引数を渡す
        outputs, _, _ = model(obs_traj_rel, grids, hidden_states, cell_states, peds_list, lookup)
        
        # 損失を計算
        loss = bivariate_loss(outputs, pred_traj_gt_rel)
        total_loss += loss.item()
        
        loss.backward()
        
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()

    avg_loss = total_loss / len(loader)
    print(f'Epoch: {epoch}, Train Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    # SocialModelが必要とする引数を追加
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5) # mu, sigma, rho
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # データと学習のパラメータ
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

    # データセットの読み込み
    all_dataset_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    train_datasets = []
    val_datasets = []
    
    print("Loading datasets...")
    for name in all_dataset_names:
        train_path = f'./datasets/{name}/train/'
        val_path = f'./datasets/{name}/val/'
        if os.path.exists(train_path):
            train_datasets.append(TrajectoryDataset(train_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1))
        if os.path.exists(val_path):
            val_datasets.append(TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1))

    dset_train = ConcatDataset(train_datasets)
    # dset_val = ConcatDataset(val_datasets) # 検証ループは簡略化のため省略

    # DataLoaderの作成 (batch_size=1, collate_fnなし)
    loader_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
    print(f"Total training sequences: {len(dset_train)}")

    # SocialModelをargsで初期化
    model = SocialModel(args).to(device)
    # 元のモデルに合わせてAdagradを使用
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    checkpoint_dir = f'./checkpoint/{args.tag}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("******************************")
    print("Training initiating....")
    for epoch in range(1, args.num_epochs + 1):
        train(epoch, model, loader_train, optimizer, args, device)
        # val_loss = vald(...) # 検証ループも同様に修正が必要

if __name__ == '__main__':
    main()
