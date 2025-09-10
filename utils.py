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
    
    if not data:
        return np.empty((0, 4))
        
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    """
    Dataloder for the Trajectory datasets
    """
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1):
        super(TrajectoryDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.min_ped = min_ped

        self.sequences = []
        all_files = sorted([os.path.join(data_dir, path) for path in os.listdir(data_dir) if path.endswith('.txt')])
        
        for path in all_files:
            # Load data for a single file
            raw_data = read_file(path)
            if raw_data.shape[0] == 0:
                continue

            # Social-LSTM format expects [frame, ped, x, y], original is [frame, ped, y, x]
            data = raw_data[:, [0, 1, 3, 2]] if raw_data.shape[1] == 4 else raw_data

            frames = np.unique(data[:, 0]).tolist()
            frame_data = {frame: data[data[:, 0] == frame, :] for frame in frames}
            
            num_frames = len(frames)
            
            # Iterate through all possible start frames
            for i in range(0, num_frames - self.seq_len + 1, self.skip):
                # Get the frames for the current sequence window
                curr_seq_frames = frames[i:i + self.seq_len]
                
                # Ensure frames have a consistent step size (handles frame skips)
                frame_diffs = np.diff(curr_seq_frames)
                if len(frame_diffs) > 0 and not np.all(frame_diffs == frame_diffs[0]):
                    continue

                # Concatenate data for the current sequence window
                curr_seq_data = np.concatenate([frame_data[f] for f in curr_seq_frames], axis=0)
                
                peds_in_seq = np.unique(curr_seq_data[:, 1])
                
                if len(peds_in_seq) < self.min_ped:
                    continue

                # Initialize the sequence array for all peds in this window
                seq_abs = np.full((len(peds_in_seq), self.seq_len, 2), np.nan)
                
                for ped_idx, ped_id in enumerate(peds_in_seq):
                    # Get data for the current pedestrian
                    ped_seq_data = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    
                    # A pedestrian must be present for the entire sequence length
                    if ped_seq_data.shape[0] != self.seq_len:
                        continue
                        
                    # Add the trajectory to the sequence array
                    seq_abs[ped_idx, :, :] = ped_seq_data[:, 2:]

                # Remove pedestrians who had missing frames (rows that still have NaNs)
                valid_ped_mask = ~np.isnan(seq_abs).any(axis=(1, 2))
                
                if np.sum(valid_ped_mask) >= self.min_ped:
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

def seq_collate(data):
    """
    Custom collate function for DataLoader to handle variable-sized sequences.
    """
    obs_traj_list, pred_traj_list, obs_traj_rel_list = zip(*data)
    
    _len = [seq.size(0) for seq in obs_traj_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    
    obs_traj = torch.cat(obs_traj_list, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_traj_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_traj_rel_list, dim=0).permute(1, 0, 2)
    
    pred_traj_rel = torch.zeros_like(pred_traj)
    pred_traj_rel[0, :, :] = pred_traj[0, :, :] - obs_traj[-1, :, :]
    pred_traj_rel[1:, :, :] = pred_traj[1:, :, :] - pred_traj[:-1, :, :]
    
    seq_start_end = torch.LongTensor(seq_start_end)
    loss_mask = torch.ones(pred_traj.size(1), pred_traj.size(0)).permute(1, 0)
    non_linear_ped = torch.zeros(pred_traj.size(1))
    
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
        non_linear_ped, loss_mask, seq_start_end
    ]
    
    return tuple(out)
