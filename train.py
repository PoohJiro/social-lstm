import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import os

# これまで作成した、書き換え済みのヘルパーファイルをインポート
from utils import TrajectoryDataset, seq_collate
from model import SocialModel
from grid import get_grid_mask
from metrics import bivariate_loss, sample_from_bivariate_gaussian, calculate_ade_fde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rel_to_abs(rel_traj, start_pos):
    """相対座標を絶対座標に変換する"""
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
        # モデルは5次元のパラメータを返す
        pred_params = model(obs_rel, grids)
        
        # 論文の損失関数（二変量ガウス分布の負の対数尤度）を使用
        loss = bivariate_loss(pred_params, pred_gt_rel)
        if torch.isnan(loss) or loss.item() == 0: continue
        
        loss.backward()
        
        # 元のコードにあった勾配クリッピングを適用
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader_train) if len(loader_train) > 0 else 0

def val_epoch(model, loader_val, args):
    """1エポック分の検証を行う"""
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
            
            pred_params = model(obs_rel, grids)
            # 評価のために、予測分布から1点をサンプリング
            pred_fake_rel = sample_from_bivariate_gaussian(pred_params)
            pred_fake_abs = rel_to_abs(pred_fake_rel, obs_abs[:, -1])
            
            ade, fde = calculate_ade_fde(pred_fake_abs, pred_gt_abs)
            if ade > 0: total_ade += ade
            if fde > 0: total_fde += fde

    avg_ade = total_ade / len(loader_val) if len(loader_val) > 0 else 0
    avg_fde = total_fde / len(loader_val) if len(loader_val) > 0 else 0
    return avg_ade, avg_fde

def main():
    parser = argparse.ArgumentParser()
    # 元のコードの引数を、現在の構造に合わせて整理
    # データ関連
    parser.add_argument('--data_path', default='./datasets', help='Path to dataset directory')
    parser.add_argument('--dataset_val', default='eth', help='Dataset to use for validation')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)

    # 学習関連
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=10.0, help='Gradient clipping')

    # モデル関連
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5, help='mux, muy, sx, sy, corr')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--neighborhood_size', type=float, default=32.0)
    parser.add_argument('--grid_size', type=int, default=4)
    args = parser.parse_args()

    # --- 1. データの準備 ---
    all_dataset_names = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
    train_paths = [os.path.join(args.data_path, name, 'train') for name in all_dataset_names]
    train_datasets = [TrajectoryDataset(path, obs_len=args.obs_len, pred_len=args.pred_len) for path in train_paths]
    
    print(f"--- Combining {len(train_datasets)} training datasets ---")
    concat_train_dataset = ConcatDataset(train_datasets)
    loader_train = DataLoader(concat_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=seq_collate)
    
    val_path = os.path.join(args.data_path, args.dataset_val, 'val')
    print(f"--- Validating on: {val_path} ---")
    dset_val = TrajectoryDataset(val_path, obs_len=args.obs_len, pred_len=args.pred_len)
    loader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=seq_collate)
    
    # --- 2. モデルとオプティマイザの準備 ---
    model = SocialModel(args).to(device)
    # 元のコードに合わせてAdagradを使用
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    
    # --- 3. 学習と検証のループ ---
    best_ade = float('inf')
    print("--- Starting Training ---")
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, loader_train, optimizer, args)
        val_ade, val_fde = val_epoch(model, loader_val, args)
        print(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {train_loss:.4f}, Val ADE: {val_ade:.4f}, Val FDE: {val_fde:.4f}")

        if val_ade < best_ade:
            best_ade = val_ade
            print(f"****** Best Val ADE so far: {best_ade:.4f}. Saving model... ******")
            torch.save(model.state_dict(), f'social_lstm_val_on_{args.dataset_val}_best.pth')

if __name__ == '__main__':
    main()
