# utils.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab': delim = '\t'
    elif delim == 'space': delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1, delim='\t'):
        super(TrajectoryDataset, self).__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.min_ped = min_ped
        self.delim = delim

        all_files = [os.path.join(self.data_dir, path) for path in os.listdir(self.data_dir) if path.endswith('.txt')]
        
        seq_list_abs = []
        for path in all_files:
            data = read_file(path, self.delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[frame == data[:, 0], :] for frame in frames]
            
            num_sequences = (len(frames) - self.seq_len + 1) // self.skip

            for idx in range(0, num_sequences * self.skip, self.skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_abs = np.zeros((len(peds_in_curr_seq), self.seq_len, 2))
                num_peds_considered = 0
                
                for ped_idx, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    if len(curr_ped_seq) != self.seq_len:
                        continue
                    
                    curr_seq_abs[num_peds_considered, :, :] = curr_ped_seq[:, 2:]
                    num_peds_considered += 1

                if num_peds_considered >= self.min_ped:
                    seq_list_abs.append(curr_seq_abs[:num_peds_considered])

        self.num_seq = len(seq_list_abs)
        self.seq_list_abs = seq_list_abs

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        abs_seq = self.seq_list_abs[index]
        
        rel_seq = np.zeros_like(abs_seq)
        rel_seq[:, 1:, :] = abs_seq[:, 1:, :] - abs_seq[:, :-1, :]
        
        obs_traj_abs = torch.from_numpy(abs_seq[:, :self.obs_len, :]).float()
        pred_traj_abs = torch.from_numpy(abs_seq[:, self.obs_len:, :]).float()
        obs_traj_rel = torch.from_numpy(rel_seq[:, :self.obs_len, :]).float()
        
        return obs_traj_abs, pred_traj_abs, obs_traj_rel

def seq_collate(data):
    (obs_abs_list, pred_abs_list, obs_rel_list) = zip(*data)

    max_peds = max([obs.shape[0] for obs in obs_abs_list])
    obs_len = obs_abs_list[0].shape[1]
    pred_len = pred_abs_list[0].shape[1]
    batch_size = len(obs_abs_list)

    obs_traj_abs = torch.full((batch_size, max_peds, obs_len, 2), float('nan'))
    pred_traj_abs = torch.full((batch_size, max_peds, pred_len, 2), float('nan'))
    obs_traj_rel = torch.full((batch_size, max_peds, obs_len, 2), float('nan'))

    for i, (obs_abs, pred_abs, obs_rel) in enumerate(zip(obs_abs_list, pred_abs_list, obs_rel_list)):
        num_peds = obs_abs.shape[0]
        obs_traj_abs[i, :num_peds, :, :] = obs_abs
        pred_traj_abs[i, :num_peds, :, :] = pred_abs
        obs_traj_rel[i, :num_peds, :, :] = obs_rel

    return obs_traj_abs.permute(0, 2, 1, 3), pred_traj_abs.permute(0, 2, 1, 3), obs_traj_rel.permute(0, 2, 1, 3)
