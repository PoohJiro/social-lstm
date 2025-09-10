import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset

def read_file(_path, delim=' '):
    """ファイルからデータを読み込み、x,yの順序を修正する"""
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if len(line) < 4: continue
            line = [float(i) for i in line]
            data.append(line)
    data = np.asarray(data)
    # 元のSocial-LSTMに合わせて [frame, ped, y, x] -> [frame, ped, x, y] に変換
    if data.shape[1] == 4:
        return data[:, [0, 1, 3, 2]]
    return data

class TrajectoryDataset(Dataset):
    """あなたのロジックをベースにしたDataloder"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1):
        super(TrajectoryDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        # ディレクトリ内の全.txtファイルを結合して処理
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path.endswith('.txt')]
        all_data = np.concatenate([read_file(path) for path in all_files], axis=0)

        # 元のコードのシーケンス作成ロジックを忠実に再現
        self.sequences = []
        frames = np.unique(all_data[:, 0]).tolist()
        frame_data = {frame: all_data[all_data[:, 0] == frame, :] for frame in frames}
        
        num_sequences = (len(frames) - self.seq_len + 1) // skip

        for idx in range(0, num_sequences * skip, skip):
            start_frame = frames[idx]
            end_frame = frames[idx + self.seq_len - 1]
            
            # フレームが連続しているか確認
            if end_frame - start_frame != self.seq_len - 1:
                continue

            curr_seq_data = np.concatenate([frame_data[f] for f in frames[idx:idx + self.seq_len]], axis=0)
            peds_in_seq = np.unique(curr_seq_data[:, 1])
            
            if len(peds_in_seq) < min_ped:
                continue
            
            # (num_peds, seq_len, 2) のテンソルを作成
            seq_abs = np.full((len(peds_in_seq), self.seq_len, 2), np.nan)
            
            # 各歩行者の軌跡を埋める
            for ped_idx, ped_id in enumerate(peds_in_seq):
                ped_seq_data = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                # シーケンスの全期間に存在しない歩行者は除外
                if ped_seq_data.shape[0] != self.seq_len:
                    continue
                seq_abs[ped_idx, :, :] = ped_seq_data[:, 2:]
            
            # 全期間存在した歩行者のみをフィルタリング
            valid_ped_mask = ~np.isnan(seq_abs).any(axis=(1, 2))
            if np.sum(valid_ped_mask) >= min_ped:
                valid_seq_abs = seq_abs[valid_ped_mask]
                valid_peds_in_seq = peds_in_seq[valid_ped_mask]
                
                # 元のコードが必要とするデータ構造を作成
                peds_list_in_seq = []
                for frame_idx in range(self.seq_len):
                    peds_list_in_seq.append(valid_peds_in_seq.tolist())

                lookup = {ped_id: i for i, ped_id in enumerate(valid_peds_in_seq)}
                
                self.sequences.append((valid_seq_abs, peds_list_in_seq, lookup))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq_abs, peds_list, lookup = self.sequences[index]
        
        # 相対座標（速度）を計算
        seq_rel = np.zeros_like(seq_abs)
        seq_rel[:, 1:, :] = seq_abs[:, 1:, :] - seq_abs[:, :-1, :]
        
        # (num_peds, seq_len, 2) -> (seq_len, num_peds, 2) に次元を変換
        obs_traj = torch.from_numpy(seq_abs[:, :self.obs_len, :]).float().permute(1, 0, 2)
        pred_traj = torch.from_numpy(seq_abs[:, self.obs_len:, :]).float().permute(1, 0, 2)
        obs_traj_rel = torch.from_numpy(seq_rel[:, :self.obs_len, :]).float().permute(1, 0, 2)

        return obs_traj, pred_traj, obs_traj_rel, peds_list, lookup

def seq_collate(data):
    """バッチ内の可変長の歩行者数をパディングして揃えるための関数"""
    (obs_traj_list, pred_traj_list, obs_traj_rel_list, peds_list_list, lookup_list) = zip(*data)

    # バッチ内の最大歩行者数を取得
    max_peds = max([obs.shape[1] for obs in obs_traj_list])
    obs_len = obs_traj_list[0].shape[0]
    pred_len = pred_traj_list[0].shape[0]
    batch_size = len(obs_traj_list)

    # パディングされたテンソルを準備
    obs_traj = torch.full((batch_size, obs_len, max_peds, 2), float('nan'))
    pred_traj = torch.full((batch_size, pred_len, max_peds, 2), float('nan'))
    obs_traj_rel = torch.full((batch_size, obs_len, max_peds, 2), float('nan'))

    # パディングされたリストを準備
    peds_list = []
    lookup = []

    for i in range(batch_size):
        num_peds = obs_traj_list[i].shape[1]
        obs_traj[i, :, :num_peds, :] = obs_traj_list[i]
        pred_traj[i, :, :num_peds, :] = pred_traj_list[i]
        obs_traj_rel[i, :, :num_peds, :] = obs_traj_rel_list[i]
        peds_list.append(peds_list_list[i])
        lookup.append(lookup_list[i])

    return obs_traj, pred_traj, obs_traj_rel, peds_list, lookup



