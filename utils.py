import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================================================
#
# HELPER FUNCTIONS (モデルの学習と評価に必要な補助関数)
#
# ===================================================================================================

def vectorize_seq(x_seq):
    """
    軌道データを絶対座標から、各歩行者の初期位置からの相対座標に変換する。
    入力: x_seq (num_peds, seq_len, 2)
    出力: 相対座標, 初期位置
    """
    first_frame_positions = x_seq[:, 0:1, :].clone()
    return x_seq - first_frame_positions, first_frame_positions

def revert_seq(x_seq_rel, first_frame_positions):
    """相対座標から絶対座標に軌道を復元する。"""
    return x_seq_rel + first_frame_positions

def getCoef(outputs):
    """モデルの出力から二変量正規分布のパラメータを抽出する"""
    # outputs shape: (seq_len, num_peds, 5)
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr

def Gaussian2DLikelihood(outputs, targets, mask):
    """
    二変量正規分布の負の対数尤度（損失）を計算する。
    mask: (seq_len, num_peds) の形状で、損失を計算すべき箇所が1になっている。
    """
    mux, muy, sx, sy, corr = getCoef(outputs)
    x_coords = targets[:, :, 0]
    y_coords = targets[:, :, 1]
    
    vx = x_coords - mux
    vy = y_coords - muy
    sx_sy = sx * sy
    epsilon = 1e-20
    
    z = (vx / sx).pow(2) + (vy / sy).pow(2) - 2 * ((corr * vx * vy) / (sx_sy))
    determinant = 1 - corr.pow(2)
    
    n = torch.log(2 * math.pi * sx_sy * torch.sqrt(determinant + epsilon)) + (z / (2 * (determinant + epsilon)))

    if torch.sum(mask) > 0:
        loss = torch.sum(n * mask) / torch.sum(mask)
    else:
        loss = torch.sum(n * mask)

    return loss

def get_mean_error(predicted_trajs, true_trajs, mask):
    """Average Displacement Error (ADE)を計算する。"""
    # shape: (pred_len, num_peds, 2)
    error = torch.sqrt((predicted_trajs[:, :, 0] - true_trajs[:, :, 0])**2 + (predicted_trajs[:, :, 1] - true_trajs[:, :, 1])**2)
    # マスクを適用し、存在する歩行者のエラーのみを平均
    error = torch.sum(error * mask) / torch.sum(mask) if torch.sum(mask) > 0 else 0
    return error.item()

def get_final_error(predicted_trajs, true_trajs, mask):
    """Final Displacement Error (FDE)を計算する。"""
    # shape: (pred_len, num_peds, 2)
    error = torch.sqrt((predicted_trajs[-1, :, 0] - true_trajs[-1, :, 0])**2 + (predicted_trajs[-1, :, 1] - true_trajs[-1, :, 1])**2)
    # 最後のフレームのマスクを適用
    mask_last_frame = mask[-1, :]
    error = torch.sum(error * mask_last_frame) / torch.sum(mask_last_frame) if torch.sum(mask_last_frame) > 0 else 0
    return error.item()

def time_lr_scheduler(optimizer, epoch, lr_decay=0.95, lr_decay_epoch=10):
    """学習率をスケジューリングする"""
    if epoch % lr_decay_epoch == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay
    return optimizer

# ===================================================================================================
#
# DATASET CLASS (Social-STGCNNベース)
#
# ===================================================================================================

class TrajectoryDataset(Dataset):
    """軌道データセットのためのDataloader"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        self.sequences = [] # 各シーケンスのデータを格納するリスト

        for path in all_files:
            # --- 変更点: delimiterをスペースまたはタブに対応するように修正 ---
            # delimiter=' ' から delimiter=None に変更することで、
            # スペースやタブなどの任意の空白文字を区切りとして認識します。
            data = np.loadtxt(path, delimiter=None, dtype=np.float32)
            
            frames = np.unique(data[:, 0]).tolist()
            frame_data = {frame: data[data[:, 0] == frame, :] for frame in frames}

            num_sequences = len(frames) - self.seq_len + 1

            for i in range(0, num_sequences, self.skip):
                # 現在のシーケンスのフレーム範囲
                current_frames = frames[i : i + self.seq_len]
                
                # シーケンス全体のデータを結合
                seq_data = np.concatenate([frame_data[f] for f in current_frames], axis=0)
                
                # このシーケンスに登場する全ての歩行者IDを取得
                peds_in_seq = np.unique(seq_data[:, 1]).astype(int)
                
                if len(peds_in_seq) < min_ped:
                    continue
                
                # (num_peds, seq_len, 2) の形状を持つテンソルを作成
                peds_data = np.zeros((len(peds_in_seq), self.seq_len, 2), dtype=np.float32)
                # マスク (歩行者がそのフレームに存在するかどうか)
                peds_mask = np.zeros((len(peds_in_seq), self.seq_len), dtype=np.float32)

                for ped_idx, ped_id in enumerate(peds_in_seq):
                    for frame_idx, frame in enumerate(current_frames):
                        ped_frame_data = frame_data[frame]
                        ped_in_frame = ped_frame_data[ped_frame_data[:, 1] == ped_id]
                        if ped_in_frame.shape[0] > 0:
                            peds_data[ped_idx, frame_idx, :] = ped_in_frame[0, [2, 3]] # x, y 座標
                            peds_mask[ped_idx, frame_idx] = 1.0

                self.sequences.append((
                    torch.from_numpy(peds_data),
                    torch.from_numpy(peds_mask)
                ))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

