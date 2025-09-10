import os
import torch
import numpy as np
from torch.utils.data import Dataset

def read_file(_path, delim='\t'):
    """ファイルからデータを読み込む。区切り文字はタブまたはスペースを想定。"""
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if not line or len(line) < 4:
                continue
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def poly_fit(traj, traj_len, threshold):
    """軌道が線形かどうかをチェックする"""
    if traj_len < 2:
        return False
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[:, 0], 2, full=True)[1]
    res_y = np.polyfit(t, traj[:, 1], 2, full=True)[1]
    return (res_x + res_y) > threshold

class TrajectoryDataset(Dataset):
    """軌道データセットのためのDataloader"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=1, delim='\t'):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.min_ped = min_ped
        self.delim = delim
        self.threshold = threshold

        all_files = [os.path.join(self.data_dir, _path) for _path in os.listdir(self.data_dir) if _path.endswith('.txt')]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        
        for path in all_files:
            data = read_file(path, self.delim)
            if data.size == 0:
                continue
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[frame == data[:, 0], :] for frame in frames]
            
            num_sequences = int(np.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip, self.skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((self.seq_len, len(peds_in_curr_seq), 2))
                curr_seq = np.zeros((self.seq_len, len(peds_in_curr_seq), 2))
                curr_loss_mask = np.zeros((self.seq_len, len(peds_in_curr_seq)))
                num_peds_considered = 0
                _non_linear_ped = []
                
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    
                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_seq = curr_ped_seq[:, 2:]
                    _idx = num_peds_considered
                    curr_seq[:self.seq_len, _idx, :] = curr_ped_seq
                    
                    rel_curr_ped_seq = np.zeros_like(curr_ped_seq)
                    rel_curr_ped_seq[1:, :] = curr_ped_seq[1:, :] - curr_ped_seq[:-1, :]
                    curr_seq_rel[:self.seq_len, _idx, :] = rel_curr_ped_seq

                    # ★★★ エラー修正箇所 ★★★
                    # poly_fit関数に渡す軌道データを観測部分(obs_len)に限定し、長さも合わせました
                    is_non_linear = poly_fit(curr_ped_seq[:self.obs_len], self.obs_len, self.threshold)
                    _non_linear_ped.append(is_non_linear)
                    curr_loss_mask[:self.seq_len, _idx] = 1
                    num_peds_considered += 1

                if num_peds_considered >= self.min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    non_linear_ped += _non_linear_ped
                    seq_list.append(curr_seq[:, :num_peds_considered, :])
                    seq_list_rel.append(curr_seq_rel[:, :num_peds_considered, :])
                    loss_mask_list.append(curr_loss_mask[:, :num_peds_considered])
        
        if not seq_list:
            self.num_seq = 0
            return

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=1)
        seq_list_rel = np.concatenate(seq_list_rel, axis=1)
        loss_mask_list = np.concatenate(loss_mask_list, axis=1)
        non_linear_ped = np.asarray(non_linear_ped)

        self.obs_traj = torch.from_numpy(seq_list[:self.obs_len, :, :]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[self.obs_len:, :, :]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:self.obs_len, :, :]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[self.obs_len:, :, :]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        
        out = [
            self.obs_traj[:, start:end, :], self.pred_traj[:, start:end, :],
            self.obs_traj_rel[:, start:end, :], self.pred_traj_rel[:, start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[:, start:end],
        ]
        
        v_obs = out[2].permute(1, 0, 2)
        v_pred = out[3].permute(1, 0, 2)
        
        num_peds = v_obs.size(0)
        
        a_obs = torch.ones(num_peds, num_peds)
        a_pred = torch.ones(num_peds, num_peds)
        
        out.append(v_obs.contiguous())
        out.append(a_obs.contiguous())
        out.append(v_pred.contiguous())
        out.append(a_pred.contiguous())
        
        return out
