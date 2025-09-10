import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import os
import time

# あなたの元のアルゴリズムファイルと、書き換えたデータローダー
from model import SocialModel
from utils import TrajectoryDataset, seq_collate
from grid import getSequenceGridMask
from helper import Gaussian2DLikelihood, get_mean_error, get_final_error, getCoef, sample_gaussian_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    # 元のコードの引数を、現在の構造に合わせて整理
    # データ関連
    parser.add_argument('--data_path', default='./datasets', help='Path to dataset directory')
    parser.add_argument('--dataset_val', default='eth', help='Dataset to use for validation')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)

    # 学習関連
    parser.add_argument('--batch_size', type=int, default=1) # 元のコードは1シーケンスずつ処理
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--dropout', type=float, default=0.0)

    # モデル関連
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5, help='mux, muy, sx, sy, corr')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0)
    parser.add_argument('--grid_size', type=int, default=4)
    args = parser.parse_args()
    args.seq_length = args.obs_len + args.pred_len
    args.use_cuda = torch.cuda.is_available()

    # --- 1. データの準備 ---
    all_dataset_names = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
    train_paths = [os.path.join(args.data_path, name, 'train') for name in all_dataset_names]
    val_path = os.path.join(args.data_path, args.dataset_val, 'val')
    
    print(f"--- Combining {len(train_paths)} training datasets ---")
    dset_train = ConcatDataset([TrajectoryDataset(path, obs_len=args.obs_len, pred_len=args.pred_len) for path in train_paths])
    loader_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
    
    print(f"--- Validating on: {val_path} ---")
    dset_val = TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len)
    loader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False)
    
    # --- 2. モデルとオプティマイザの準備 ---
    net = SocialModel(args).to(device)
    # 元のコードに合わせてAdagradを使用
    optimizer = torch.optim.Adagrad(net.parameters(), lr=args.lr)
    
    # --- 3. 学習と検証のループ ---
    best_ade = float('inf')
    for epoch in range(args.num_epochs):
        train(epoch, net, loader_train, optimizer, args)
        val_ade, val_fde = validate(epoch, net, loader_val, args)
        print(f"Epoch {epoch+1}/{args.num_epochs} -> Val ADE: {val_ade:.4f}, Val FDE: {val_fde:.4f}")

        if val_ade < best_ade:
            best_ade = val_ade
            print(f"****** Best Val ADE: {best_ade:.4f}. Saving model... ******")
            torch.save(net.state_dict(), f'social_lstm_val_on_{args.dataset_val}_best.pth')

def train(epoch, net, dataloader, optimizer, args):
    """元のtrain関数のロジックを再現"""
    net.train()
    total_loss = 0
    for batch in dataloader:
        # 新しいDataLoaderからの出力を受け取る
        obs_traj, pred_traj_gt, obs_traj_rel, peds_list, lookup = batch
        
        # バッチサイズ1なのでsqueeze(0)で次元を削除
        obs_traj, pred_traj_gt, obs_traj_rel = [t.squeeze(0).to(device) for t in [obs_traj, pred_traj_gt, obs_traj_rel]]
        lookup = {k.item(): v.item() for k,v in lookup[0].items()} if isinstance(lookup, list) else lookup

        num_peds = obs_traj.size(1)
        
        grids = [getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size) for t in range(args.obs_len)]
        hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
        cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
            
        optimizer.zero_grad()
        
        # 元のmodel.pyのforwardに必要な引数を渡す
        outputs, _, _ = net(obs_traj_rel, grids, hidden_states, cell_states, peds_list, lookup)
        
        # 正解の相対座標を計算
        pred_traj_gt_rel = torch.zeros_like(pred_traj_gt)
        pred_traj_gt_rel[0] = pred_traj_gt[0] - obs_traj[-1]
        pred_traj_gt_rel[1:] = pred_traj_gt[1:] - pred_traj_gt[:-1]

        # 元のhelper.pyの損失関数を使用
        loss = Gaussian2DLikelihood(outputs, pred_traj_gt_rel, peds_list[args.obs_len:], lookup)
        if torch.isnan(loss) or loss.item() == 0: continue

        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += loss.item()

    print(f"TRAIN:\t Epoch: {epoch}\t Loss: {total_loss / len(dataloader):.4f}")

def validate(epoch, net, dataloader, args):
    """元のvalidationロジックを再現"""
    net.eval()
    total_ade, total_fde = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            obs_traj, pred_traj_gt, obs_traj_rel, peds_list, lookup = batch
            obs_traj, pred_traj_gt, obs_traj_rel = [t.squeeze(0).to(device) for t in [obs_traj, pred_traj_gt, obs_traj_rel]]
            lookup = {k.item(): v.item() for k,v in lookup[0].items()} if isinstance(lookup, list) else lookup

            num_peds = obs_traj.size(1)
            grids = [getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size) for t in range(args.obs_len)]
            hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
            cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
            
            # 予測
            outputs, _, _ = net(obs_traj_rel, grids, hidden_states, cell_states, peds_list, lookup)
            
            # ADE/FDEを計算
            mux, muy, sx, sy, corr = getCoef(outputs)
            pred_traj_fake_rel = torch.stack([mux, muy], dim=-1)
            pred_traj_fake_abs = torch.cumsum(pred_traj_fake_rel, dim=0) + obs_traj[-1]
            
            peds_present_at_pred = peds_list[args.obs_len:]
            ade = get_mean_error(pred_traj_fake_abs, pred_traj_gt, peds_present_at_pred, peds_present_at_pred, args.use_cuda, lookup)
            fde = get_final_error(pred_traj_fake_abs, pred_traj_gt, peds_present_at_pred, peds_present_at_pred, lookup)
            
            if not torch.isnan(ade): total_ade += ade.item()
            if not torch.isnan(fde): total_fde += fde.item()
            
    avg_ade = total_ade / len(dataloader)
    avg_fde = total_fde / len(dataloader)
    return avg_ade, avg_fde

if __name__ == '__main__':
    main()
