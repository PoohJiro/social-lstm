import os
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import argparse
import pickle

# Social-LSTMのモデル
from model import SocialModel
# Social-STGCNNのutils.pyをそのまま使用
from utils import TrajectoryDataset
from metrics import bivariate_loss

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

        # STGCNNのTrajectoryDatasetは10個の要素を返す
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        # デバイスに移動
        obs_traj = obs_traj.to(device)
        pred_traj_gt = pred_traj_gt.to(device)
        obs_traj_rel = obs_traj_rel.to(device)
        pred_traj_gt_rel = pred_traj_gt_rel.to(device)
        loss_mask = loss_mask.to(device)
        V_obs = V_obs.to(device)
        V_tr = V_tr.to(device)

        # データ形式を調整 (batch, feat, seq) -> (seq, batch, feat)
        obs_traj = obs_traj.squeeze(0).permute(2, 0, 1)  # (seq, peds, 2)
        obs_traj_rel = obs_traj_rel.squeeze(0).permute(2, 0, 1)
        pred_traj_gt = pred_traj_gt.squeeze(0).permute(2, 0, 1)
        pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).permute(2, 0, 1)
        
        num_peds = obs_traj.size(1)
        
        # グリッドマスクを作成
        grids = []
        for t in range(args.obs_seq_len):
            grid = getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size)
            grids.append(grid.to(device))
        
        # 隠れ状態を初期化
        hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
        cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
        
        # ペデストリアンリストとルックアップを作成
        peds_list = [torch.arange(num_peds).to(device) for _ in range(args.obs_seq_len + args.pred_seq_len)]
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
                print(f'TRAIN:\t Epoch: {epoch}\t Loss: {loss_batch/batch_count:.4f}')
                
        except Exception as e:
            print(f"Error in training batch: {e}")
            continue
            
    return loss_batch/batch_count if batch_count > 0 else 0

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
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                
            # デバイスに移動
            obs_traj = obs_traj.to(device)
            pred_traj_gt = pred_traj_gt.to(device)
            obs_traj_rel = obs_traj_rel.to(device)
            pred_traj_gt_rel = pred_traj_gt_rel.to(device)
            
            # データ形式を調整
            obs_traj = obs_traj.squeeze(0).permute(2, 0, 1)
            obs_traj_rel = obs_traj_rel.squeeze(0).permute(2, 0, 1)
            pred_traj_gt = pred_traj_gt.squeeze(0).permute(2, 0, 1)
            pred_traj_gt_rel = pred_traj_gt_rel.squeeze(0).permute(2, 0, 1)
            
            num_peds = obs_traj.size(1)
            
            # グリッドマスクを作成
            grids = []
            for t in range(args.obs_seq_len):
                grid = getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size)
                grids.append(grid.to(device))
            
            # 隠れ状態を初期化
            hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
            cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
            
            peds_list = [torch.arange(num_peds).to(device) for _ in range(args.obs_seq_len + args.pred_seq_len)]
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
                    print(f'VALD:\t Epoch: {epoch}\t Loss: {loss_batch/batch_count:.4f}')
                    
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
                
    return loss_batch/batch_count if batch_count > 0 else 0

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
    
    # Data specific parameters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='all',
                        help='eth,hotel,univ,zara1,zara2,all')    

    # Training specific parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=10.0,
                        help='gradient clipping')        
    parser.add_argument('--lr', type=float, default=0.003,
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

    # Data prep
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    
    # データセットの選択
    all_dataset_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    
    if args.dataset == 'all':
        # 全データセットで学習
        train_datasets = []
        val_datasets = []
        
        print("Loading all datasets for training...")
        for dataset_name in all_dataset_names:
            train_path = f'./datasets/{dataset_name}/train/'
            val_path = f'./datasets/{dataset_name}/val/'
            
            if os.path.exists(train_path):
                # norm_lap_matrパラメータを削除
                train_datasets.append(
                    TrajectoryDataset(
                        train_path,
                        obs_len=obs_seq_len,
                        pred_len=pred_seq_len,
                        skip=1
                        # norm_lap_matr=True を削除
                    )
                )
                print(f"  - Loaded {dataset_name} training data")
            
            if os.path.exists(val_path):
                val_datasets.append(
                    TrajectoryDataset(
                        val_path,
                        obs_len=obs_seq_len,
                        pred_len=pred_seq_len,
                        skip=1
                        # norm_lap_matr=True を削除
                    )
                )
        
        # 全データセットを結合
        dset_train = ConcatDataset(train_datasets) if train_datasets else train_datasets[0]
        dset_val = ConcatDataset(val_datasets) if val_datasets else val_datasets[0]
        print(f"Total training sequences: {len(dset_train)}")
        print(f"Total validation sequences: {len(dset_val)}")
    else:
        # 特定のデータセットのみ
        data_set = f'./datasets/{args.dataset}/'
        
        dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1
            # norm_lap_matr=True を削除
        )
    
    dset_val = TrajectoryDataset(
        data_set + 'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1
        # norm_lap_matr=True を削除
    )
        
        dset_val = TrajectoryDataset(
            data_set + 'val/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,
            norm_lap_matr=True
        )
        
        print(f"Dataset: {args.dataset}")
        print(f"Training sequences: {len(dset_train)}")
        print(f"Validation sequences: {len(dset_val)}")

    # DataLoader作成
    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # STGCNNと同じく1固定
        shuffle=True,
        num_workers=0
    )

    loader_val = DataLoader(
        dset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Social-LSTMモデルを定義
    model = SocialModel(args).to(device)
    print(f"Model initialized on {device}")

    # Optimizer設定
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)  # Social-LSTMはAdagradを使用

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    # チェックポイントディレクトリ
    checkpoint_dir = f'./checkpoint/{args.tag}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # Training
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

        # ベストモデルの保存
        if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')
            print(f"  *** New best model saved at epoch {epoch} ***")

        # エポック毎の結果表示
        print('*'*30)
        print(f'Epoch: {epoch}/{args.num_epochs}')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Best Val Loss: {constant_metrics["min_val_loss"]:.4f} (Epoch {constant_metrics["min_val_epoch"]})')
        print('*'*30)
        
        # メトリクスの保存
        with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)
        
        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)

    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_dir}val_best.pth")
    print(f"Best validation loss: {constant_metrics['min_val_loss']:.4f}")

if __name__ == '__main__':
    main()
