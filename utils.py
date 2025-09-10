import os
import torch
import numpy as np
from torch.utils.data import Dataset

def read_file(_path, delim=' '):
    """ファイルからデータを読み込む。空のファイルに対応。"""
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if len(line) < 4: continue
            line = [float(i) for i in line]
            data.append(line)
    
    # ★★★ バグ修正箇所 ★★★
    # ファイルが空だった場合、正しい形状の空の配列を返す
    if not data:
        return np.empty((0, 4))
        
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    """あなたのロジックをベースにしたDataloder"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1):
        super(TrajectoryDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        # ★★★ 修正箇所 ★★★
        # ディレクトリ内の.txtファイルを名前順で読み込む
        all_files = sorted([os.path.join(data_dir, path) for path in os.listdir(data_dir) if path.endswith('.txt')])
        
        all_data_list = []
        for path in all_files:
            raw_data = read_file(path)
            # ファイルが空でなければリストに追加
            if raw_data.shape[0] > 0:
                # 元のSocial-LSTMに合わせて [frame, ped, y, x] -> [frame, ped, x, y] に変換
                data = raw_data[:, [0, 1, 3, 2]] if raw_data.shape[1] == 4 else raw_data
                all_data_list.append(data)
        
        # データが一つも読み込めなかった場合のエラー処理
        if not all_data_list:
            self.sequences = []
            return

        cumulative_data = np.concatenate(all_data_list, axis=0)

        # 元のコードのシーケンス作成ロジックを忠実に再現
        self.sequences = []
        frames = np.unique(cumulative_data[:, 0]).tolist()
        frame_data = {frame: cumulative_data[cumulative_data[:, 0] == frame, :] for frame in frames}
        
        num_sequences = (len(frames) - self.seq_len + 1) // skip

        for idx in range(0, num_sequences * skip, skip):
            start_frame = frames[idx]
            end_frame = frames[idx + self.seq_len - 1]
            
            if end_frame - start_frame != self.seq_len - 1:
                continue

            curr_seq_data = np.concatenate([frame_data[f] for f in frames[idx:idx + self.seq_len]], axis=0)
            peds_in_seq = np.unique(curr_seq_data[:, 1])
            
            if len(peds_in_seq) < min_ped:
                continue
            
            seq_abs = np.full((len(peds_in_seq), self.seq_len, 2), np.nan)
            
            for ped_idx, ped_id in enumerate(peds_in_seq):
                ped_seq_data = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                if ped_seq_data.shape[0] != self.seq_len:
                    continue
                seq_abs[ped_idx, :, :] = ped_seq_data[:, 2:]
            
            valid_ped_mask = ~np.isnan(seq_abs).any(axis=(1, 2))
            if np.sum(valid_ped_mask) >= min_ped:
                self.sequences.append(seq_abs[valid_ped_mask])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        abs_seq = self.sequences[index]
        rel_seq = np.zeros_like(abs_seq)
        rel_seq[:, 1:, :] = abs_seq[:, 1:, :] - abs_seq[:, :-1, :]
        
        obs_traj_abs = torch.from_numpy(abs_seq[:, :self.obs_len, :]).float()
        pred_traj_abs = torch.from_numpy(abs_seq[:, self.obs_len:, :]).float()
        obs_traj_rel = torch.from_numpy(rel_seq[:, :self.obs_len, :]).float()
        
        return obs_traj_abs, pred_traj_abs, obs_traj_rel



