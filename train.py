import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pickle

# 既存の互換性のあるモジュールをインポート
from utils import TrajectoryDataset, seq_collate
from model import SocialLSTM

def train(epoch, model, loader, optimizer, args, device):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        # seq_collateから返される7つの値を受け取る
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end) = batch

        # テンソルを適切なデバイスに移動
        obs_traj_rel = obs_traj_rel.to(device)
        pred_traj_gt_rel = pred_traj_gt_rel.to(device)
        loss_mask = loss_mask.to(device)
        seq_start_end = seq_start_end.to(device)

        optimizer.zero_grad()
        
        pred_traj_fake_rel = model(obs_traj_rel, seq_start_end)
        
        # L2損失を計算
        loss = torch.norm(pred_traj_fake_rel - pred_traj_gt_rel, p=2, dim=2)
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        loss.backward()
        
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f'Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s')
    return avg_loss

def vald(epoch, model, loader, args, device):
    """検証データでモデルを評価する関数"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            obs_traj_rel = obs_traj_rel.to(device)
            pred_traj_gt_rel = pred_traj_gt_rel.to(device)
            loss_mask = loss_mask.to(device)
            seq_start_end = seq_start_end.to(device)

            pred_traj_fake_rel = model(obs_traj_rel, seq_start_end)

            loss = torch.norm(pred_traj_fake_rel - pred_traj_gt_rel, p=2, dim=2)
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f'Epoch: {epoch}, Val Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    # 引数を設定
    parser = argparse.ArgumentParser(description='Social-LSTM')
    
    # モデルのパラメータ
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    
    # データのパラメータ
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--dataset', default='all', help='eth,hotel,univ,zara1,zara2,all')
    
    # 学習のパラメータ
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10, help='number of epochs to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use learning rate scheduler')
    parser.add_argument('--tag', default='social_lstm', help='personal tag for the model')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 複数データセットの読み込みロジック
    all_dataset_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    train_datasets = []
    val_datasets = []

    if args.dataset == 'all':
        print("Loading all datasets...")
        for dataset_name in all_dataset_names:
            train_path = f'./datasets/{dataset_name}/train/'
            val_path = f'./datasets/{dataset_name}/val/'
            if os.path.exists(train_path):
                train_datasets.append(TrajectoryDataset(train_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1))
                print(f"  - Loaded {dataset_name} training data")
            if os.path.exists(val_path):
                val_datasets.append(TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1))
                print(f"  - Loaded {dataset_name} validation data")
        dset_train = ConcatDataset(train_datasets)
        dset_val = ConcatDataset(val_datasets)
    else:
        print(f"Loading dataset: {args.dataset}")
        train_path = f'./datasets/{args.dataset}/train/'
        val_path = f'./datasets/{args.dataset}/val/'
        dset_train = TrajectoryDataset(train_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1)
        dset_val = TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1)

    # seq_collateを指定してDataLoaderを作成
    loader_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=seq_collate)
    loader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=seq_collate)
    print(f"Total training sequences: {len(dset_train)}")
    print(f"Total validation sequences: {len(dset_val)}")

    # モデル、オプティマイザ、スケジューラを定義
    model = SocialLSTM(embedding_dim=args.embedding_size, h_dim=args.rnn_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.5)

    # チェックポイントとメトリクスの設定
    checkpoint_dir = f'./checkpoint/{args.tag}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, 'args.pkl'), 'wb') as fp:
        pickle.dump(args, fp)
    
    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)
    
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': float('inf')}

    # メインの学習ループ
    print("******************************")
    print("Training initiating....")
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(epoch, model, loader_train, optimizer, args, device)
        val_loss = vald(epoch, model, loader_val, args, device)
        
        if args.use_lrschd:
            scheduler.step()

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)

        # 検証ロスが最小のモデルを保存
        if val_loss < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = val_loss
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'val_best.pth'))
            print(f"  *** New best model saved at epoch {epoch} ***")

        # エポックごとにメトリクスを保存
        with open(os.path.join(checkpoint_dir, 'metrics.pkl'), 'wb') as fp:
            pickle.dump(metrics, fp)
        with open(os.path.join(checkpoint_dir, 'constant_metrics.pkl'), 'wb') as fp:
            pickle.dump(constant_metrics, fp)
            
        print('-'*30)

    print("\nTraining completed!")
    print(f"Best model saved at: {os.path.join(checkpoint_dir, 'val_best.pth')}")
    print(f"Best validation loss: {constant_metrics['min_val_loss']:.4f} at epoch {constant_metrics['min_val_epoch']}")

if __name__ == '__main__':
    main()
