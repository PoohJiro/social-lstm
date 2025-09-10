import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset

def read_file(_path, delim=' '):
    """ファイルからデータを読み込む"""
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            # 空行や不正な行をスキップ
            if len(line) < 4: continue
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    """あなたのロジックをベースにしたDataloder"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1):
        super(TrajectoryDataset, self).__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len

        # ディレクトリ内の全.txtファイルを結合して処理
        all_files = [os.path.join(self.data_dir, path) for path in os.listdir(self.data_dir) if path.endswith('.txt')]
        
        # 1つのデータセットとして全データを結合
        all_data = []
        for path in all_files:
            # 元のSocial-LSTMはy,x順なので、x,y順に修正
            raw_data = read_file(path)
            data = raw_data[:, [0, 1, 3, 2]] if raw_data.shape[1] == 4 else raw_data
            all_data.append(data)
        
        cumulative_data = np.concatenate(all_data, axis=0)

        # 元のコードのシーケンス作成ロジックを忠実に再現
        seq_list = []
        frames = np.unique(cumulative_data[:, 0]).tolist()
        frame_data = {frame: cumulative_data[cumulative_data[:, 0] == frame, :] for frame in frames}
        
        num_sequences = (len(frames) - self.seq_len + 1) // self.skip

        for idx in range(0, num_sequences * self.skip, self.skip):
            start_frame = frames[idx]
            end_frame = frames[idx + self.seq_len - 1]
            
            # フレームが連続しているか確認 (Social-STGCNNデータセットは飛び飛びの場合がある)
            if end_frame - start_frame != self.seq_len - 1:
                continue

            curr_seq_data = np.concatenate([frame_data[f] for f in frames[idx:idx + self.seq_len]], axis=0)
            peds_in_seq = np.unique(curr_seq_data[:, 1])
            
            if len(peds_in_seq) < min_ped:
                continue
            
            # (num_peds, seq_len, 2) のテンソルを作成
            curr_seq_abs = np.full((len(peds_in_seq), self.seq_len, 2), np.nan)
            
            # 各歩行者の軌跡を埋める
            for ped_idx, ped_id in enumerate(peds_in_seq):
                ped_seq_data = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                # シーケンスの途中で現れたり消えたりする歩行者を考慮
                start_t = int(ped_seq_data[0, 0] - start_frame)
                end_t = int(ped_seq_data[-1, 0] - start_frame) + 1
                
                # シーケンスの全期間に存在しない歩行者は除外
                if ped_seq_data.shape[0] != self.seq_len:
                    continue
                    
                curr_seq_abs[ped_idx, :, :] = ped_seq_data[:, 2:]
            
            # 全期間存在した歩行者のみをフィルタリング
            valid_ped_mask = ~np.isnan(curr_seq_abs).any(axis=(1, 2))
            if np.sum(valid_ped_mask) >= min_ped:
                seq_list.append(curr_seq_abs[valid_ped_mask])

        self.num_seq = len(seq_list)
        self.seq_list_abs = seq_list

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        # 絶対座標のシーケンスを取得
        seq_abs = self.seq_list_abs[index]
        
        # 相対座標（速度）を計算
        seq_rel = np.zeros_like(seq_abs)
        seq_rel[:, 1:, :] = seq_abs[:, 1:, :] - seq_abs[:, :-1, :]
        
        # (num_peds, seq_len, 2) -> (seq_len, num_peds, 2) に次元を変換
        obs_traj = torch.from_numpy(seq_abs[:, :self.obs_len, :]).float().permute(1, 0, 2)
        pred_traj = torch.from_numpy(seq_abs[:, self.obs_len:, :]).float().permute(1, 0, 2)
        obs_traj_rel = torch.from_numpy(seq_rel[:, :self.obs_len, :]).float().permute(1, 0, 2)

        # 元のコードの出力形式に合わせて、各シーケンスの歩行者リストとルックアップテーブルも返す
        peds_in_seq = np.arange(seq_abs.shape[0])
        peds_list_in_seq = [peds_in_seq.tolist()] * self.seq_len
        lookup = {i: i for i in peds_in_seq}

        return obs_traj, pred_traj, obs_traj_rel, peds_list_in_seq, lookup



