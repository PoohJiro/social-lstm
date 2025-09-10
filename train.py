import os
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import pickle

# Social-LSTMのモデルとユーティリティ
from model import SocialModel
from utils import TrajectoryDataset  # STGCNNのTrajectoryDatasetを使用
from metrics import *  # STGCNNのmetricsを使用

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getSequenceGridMask(sequence, neighborhood_size, grid_size):
    """
    Get the grid mask for a given sequence
    sequence: (num_peds, 2) tensor
    """
    num_peds = sequence.size(0)
    grid_mask = torch.zeros(num_peds, num_peds, grid_size * grid_size)
    
    for i in range(num_peds):
        for j in range(num_peds):
            if i == j:
                continue
            rel_pos = sequence[j] - sequence[i]
            cell_x = int((rel_pos[0] + neighborhood_size) / (2 * neighborhood_size / grid_size))
            cell_y = int((rel_pos[1] + neighborhood_size) / (2 * neighborhood_size / grid_size))
            
            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                cell_idx = cell_y * grid_size + cell_x
                grid_mask[i, j, cell_idx] = 1
                
    return grid_mask

def train(epoch, model, loader_train, optimizer, args, device):
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len/args.batch_size)*args.batch_size + loader_len%args.batch_size - 1

    for cnt, batch in enumerate(loader_train): 
        batch_count += 1

        # STGCNNデータローダーの出力形式に対応
        # batch = [tensor.to(device) for tensor in batch]
        # STGCNNは10個の要素を返す
        if len(batch) == 10:
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch
        else:
            # 古い形式のデータローダーの場合
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
            loss_mask, seq_start_end = batch
            # V_obs, A_obsを作成（ダミー）
            batch_size = obs_traj.size(0)
            num_peds = obs_traj.size(2)
            V_obs = torch.zeros(batch_size, args.obs_seq_len, num_peds, 2)
            A_obs = torch.zeros(batch_size, args.obs_seq_len, num_peds, num_peds)
            V_tr = pred_traj_gt
            A_tr = torch.zeros_like(A_obs)

        # デバイスに移動
        obs_traj = obs_traj.to(device)
        pred_traj_gt = pred_traj_gt.to(device)
        obs_traj_rel = obs_traj_rel.to(device)
        pred_traj_gt_rel = pred_traj_gt_rel.to(device)
        loss_mask = loss_mask.to(device)

        # Social-LSTM用の入力を準備
        # obs_traj: (obs_len, batch, 2) の形式に変換
        if obs_traj.dim() == 4:  # (batch, seq, node, feat)の場合
            obs_traj = obs_traj.squeeze(0).permute(0, 2, 1)  # (seq, node, feat)
            obs_traj_rel = obs_traj_rel.squeeze(0).permute(0, 2, 1)
            pred_traj_gt = pred_traj_gt.squeeze(0).permute(0, 2, 1)
            pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).permute(0, 2, 1)
        
        num_peds = obs_traj.size(1) if obs_traj.dim() == 3 else obs_traj.size(0)
        
        # グリッドマスクを作成
        grids = []
        for t in range(args.obs_seq_len):
            if obs_traj.dim() == 3:
                grid = getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size)
            else:
                grid = getSequenceGridMask(obs_traj, args.neighborhood_size, args.grid_size)
            grids.append(grid.to(device))
        
        # 隠れ状態を初期化
        hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
        cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
        
        # ペデストリアンリストとルックアップを作成
        peds_list = [torch.arange(num_peds) for _ in range(args.obs_seq_len + args.pred_seq_len)]
        lookup = {i: i for i in range(num_peds)}

        optimizer.zero_grad()
        
        try:
            # Social-LSTMのフォワードパス
            outputs, _, _ = model(obs_traj_rel, grids, hidden_states, cell_states, peds_list, lookup)
            
            # 損失を計算 (bivariate_lossを使用)
            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = bivariate_loss(outputs[:, :, :2], pred_traj_gt_rel[:, :, :2])
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l
            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                loss.backward()
                
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                optimizer.step()
                loss_batch += loss.item()
                print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch/batch_count)
                
        except Exception as e:
            print(f"Error in training batch: {e}")
            continue
            
    return loss_batch/batch_count

def vald(epoch, model, loader_val, args, device):
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len/args.batch_size)*args.batch_size + loader_len%args.batch_size - 1
    
    with torch.no_grad():
        for cnt, batch in enumerate(loader_val): 
            batch_count += 1

            # データを取得
            if len(batch) == 10:
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch
            else:
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
                loss_mask, seq_start_end = batch
                
            # デバイスに移動
            obs_traj = obs_traj.to(device)
            pred_traj_gt = pred_traj_gt.to(device)
            obs_traj_rel = obs_traj_rel.to(device)
            pred_traj_gt_rel = pred_traj_gt_rel.to(device)
            
            # データ形式を調整
            if obs_traj.dim() == 4:
                obs_traj = obs_traj.squeeze(0).permute(0, 2, 1)
                obs_traj_rel = obs_traj_rel.squeeze(0).permute(0, 2, 1)
                pred_traj_gt = pred_traj_gt.squeeze(0).permute(0, 2, 1)
                pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).permute(0, 2, 1)
            
            num_peds = obs_traj.size(1) if obs_traj.dim() == 3 else obs_traj.size(0)
            
            # グリッドマスクを作成
            grids = []
            for t in range(args.obs_seq_len):
                if obs_traj.dim() == 3:
                    grid = getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size)
                else:
                    grid = getSequenceGridMask(obs_traj, args.neighborhood_size, args.grid_size)
                grids.append(grid.to(device))
            
            # 隠れ状態を初期化
            hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
            cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
            
            peds_list = [torch.arange(num_peds) for _ in range(args.obs_seq_len + args.pred_seq_len)]
            lookup = {i: i for i in range(num_peds)}
            
            try:
                # 予測
                outputs, _, _ = model(obs_traj_rel, grids, hidden_states, cell_states, peds_list, lookup)
                
                # 損失を計算
                if batch_count % args.batch_size != 0 and cnt != turn_point:
                    l = bivariate_loss(outputs[:, :, :2], pred_traj_gt_rel[:, :, :2])
                    if is_fst_loss:
                        loss = l
                        is_fst_loss = False
                    else:
                        loss += l
                else:
                    loss = loss / args.batch_size
                    is_fst_loss = True
                    loss_batch += loss.item()
                    print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch/batch_count)
                    
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
                
    return loss_batch/batch_count

def main():
    parser = argparse.ArgumentParser()

    # Social-LSTM specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)  # Gaussian parameters
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Data specific parameters (STGCNNと同じ)
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')    

    # Training specific parameters (STGCNNと同じ)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gradient clipping')        
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')  
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='social_lstm',
                        help='personal tag for the model')
                        
    args = parser.parse_args()
    args.seq_length = args.obs_seq_len + args.pred_seq_len
    args.use_cuda = torch.cuda.is_available()

    print('*'*30)
    print("Training initiating....")
    print(args)

    # Data prep (STGCNNと同じ形式)    
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    data_set = './datasets/' + args.dataset + '/'

    # STGCNNのTrajectoryDatasetを使用
    dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,
            norm_lap_matr=True)  # STGCNNのパラメータ

    loader_train = DataLoader(
            dset_train,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=True,
            num_workers=0)

    dset_val = TrajectoryDataset(
            data_set + 'val/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,
            norm_lap_matr=True)

    loader_val = DataLoader(
            dset_val,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=False,
            num_workers=0)

    # Social-LSTMモデルを定義
    model = SocialModel(args).to(device)

    # Training settings (STGCNNと同じ)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    checkpoint_dir = './checkpoint/' + args.tag + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # Training (STGCNNと同じ形式)
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    print('Training started ...')
    for epoch in range(args.num_epochs):
        train_loss = train(epoch, model, loader_train, optimizer, args, device)
        val_loss = vald(epoch, model, loader_val, args, device)
        
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        
        if args.use_lrschd:
            scheduler.step()

        if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')

        print('*'*30)
        print('Epoch:', args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*'*30)
        
        with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)
        
        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)

if __name__ == '__main__':
    # マルチプロセシング対応
    import multiprocessing
    multiprocessing.freeze_support()
    main()
