# utils.py (Social-STGCNNの構造をSocial-LSTM用に改造した最終版)
import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def read_file(_path, delim='\t'):
    """指定されたパスからデータを読み込むヘルパー関数"""
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    """Trajectoryデータセット用のPyTorch Datasetクラス"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: データセットファイルが含まれるディレクトリ
        - obs_len: 観測期間の長さ
        - pred_len: 予測期間の長さ
        - skip: シーケンスを作成する際のフレームのスキップ数
        - min_ped: シーケンスに含まれるべき最小の歩行者数
        - delim: データファイルの区切り文字
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.min_ped = min_ped

        all_files = [os.path.join(self.data_dir, path) for path in os.listdir(self.data_dir)]
        
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []

        for path in all_files:
            data = read_file(path, self.delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[frame == data[:, 0], :] for frame in frames]
            
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            for idx in range(0, num_sequences * self.skip, self.skip):
                # 現在のシーケンスのデータを取得
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                
                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    
                    # シーケンスの長さに満たない歩行者は除外
                    if len(curr_ped_seq) != self.seq_len:
                        continue
                    
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    
                    # 相対座標（速度）を計算
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    
                    _idx = num_peds_considered
                    curr_seq[_idx, :, :] = curr_ped_seq
                    curr_seq_rel[_idx, :, :] = rel_curr_ped_seq
                    num_peds_considered += 1

                if num_peds_considered >= self.min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        
        # ★★★ Social-LSTMではグラフは不要なため、グラフ作成処理は全て削除 ★★★
        #     seq_to_graph や V_obs, A_obs といった変数は使いません。
        
        # 代わりに、絶対座標と相対座標のシーケンスをそのまま保持します。
        self.seq_list_abs = seq_list
        self.seq_list_rel = seq_list_rel

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        # Social-LSTMモデルが必要とするデータのみを返すように修正
        obs_traj_abs = torch.from_numpy(self.seq_list_abs[index][:, :, :self.obs_len]).float()
        pred_traj_abs = torch.from_numpy(self.seq_list_abs[index][:, :, self.obs_len:]).float()
        obs_traj_rel = torch.from_numpy(self.seq_list_rel[index][:, :, :self.obs_len]).float()
        pred_traj_rel = torch.from_numpy(self.seq_list_rel[index][:, :, self.obs_len:]).float()
        
        return (
            obs_traj_abs.permute(2, 0, 1),   # (obs_len, num_peds, 2)
            pred_traj_abs.permute(2, 0, 1),  # (pred_len, num_peds, 2)
            obs_traj_rel.permute(2, 0, 1),   # (obs_len, num_peds, 2)
            pred_traj_rel.permute(2, 0, 1),  # (pred_len, num_peds, 2)
        )
