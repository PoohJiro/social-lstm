# utils.py
import os
import pickle
import numpy as np
import torch

class DataLoader:
    def __init__(self, data_dir, batch_size=50, obs_len=8, pred_len=12, forcePreProcess=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        self.dataset_name = os.path.basename(data_dir)
        # Social-STGCNNのデータセットは通常 .txt 形式
        data_file = os.path.join(self.data_dir, f"{self.dataset_name}.txt")
        processed_file = os.path.join(self.data_dir, "trajectories.cpkl")

        if not os.path.exists(processed_file) or forcePreProcess:
            print(f"Creating pre-processed data for {self.dataset_name}...")
            self.frame_preprocess(data_file, processed_file)

        self.load_preprocessed(processed_file)
        self.reset_batch_pointer()

    def frame_preprocess(self, data_file, out_file):
        # 軌跡データをフレームID、歩行者IDでグループ化
        data = np.loadtxt(data_file, delimiter='\t')
        frames = np.unique(data[:, 0]).tolist()
        
        frame_data = []
        for frame in frames:
            frame_data.append(data[data[:, 0] == frame, :])
        
        # 歩行者IDごとの軌跡に変換 (IDは浮動小数点数から整数へ)
        peds = np.unique(data[:, 1]).astype(int)
        num_peds = np.max(peds)
        
        traj_data = np.full((len(frames), num_peds + 1, 4), np.nan) # nanで初期化

        for idx, frame in enumerate(frame_data):
            for ped in frame:
                traj_data[idx, int(ped[1]), :] = ped
        
        with open(out_file, "wb") as f:
            pickle.dump(traj_data, f)

    def load_preprocessed(self, data_file):
        with open(data_file, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.sequences_abs = [] # 絶対座標

        # self.seq_len の長さを持つシーケンスを切り出す
        for i in range(self.raw_data.shape[0] - self.seq_len + 1):
            seq_abs_all_peds = self.raw_data[i:i+self.seq_len, :, 2:4].copy()
            
            # このシーケンスに一度でも登場する歩行者の列のみを抽出
            peds_present_mask = ~np.all(np.isnan(seq_abs_all_peds), axis=(0, 2))
            
            # 歩行者が一人もいないシーケンスはスキップ
            if not np.any(peds_present_mask):
                continue
            
            seq_abs = seq_abs_all_peds[:, peds_present_mask, :]
            
            # シーケンスの開始時点で存在しない歩行者の軌跡を線形補間で埋める
            for p in range(seq_abs.shape[1]):
                if np.isnan(seq_abs[0, p, 0]):
                    # 最初に現れるフレームを探す
                    first_valid_idx = np.where(~np.isnan(seq_abs[:, p, 0]))[0]
                    if len(first_valid_idx) > 0:
                        first_idx = first_valid_idx[0]
                        seq_abs[:first_idx, p, :] = seq_abs[first_idx, p, :]
            
            self.sequences_abs.append(seq_abs)

        self.num_sequences = len(self.sequences_abs)
        self.num_batches = int(self.num_sequences / self.batch_size)
        print(f"Loaded {self.num_sequences} sequences from {self.dataset_name}, creating {self.num_batches} batches.")

    def next_batch(self):
        if self.pointer + self.batch_size > self.num_sequences:
            self.reset_batch_pointer()

        start = self.pointer
        end = self.pointer + self.batch_size
        indices_in_batch = self.indices[start:end]
        self.pointer += self.batch_size
        
        max_peds = 0
        for i in indices_in_batch:
            max_peds = max(max_peds, self.sequences_abs[i].shape[1])
        
        batch_obs_abs = np.full((self.batch_size, self.obs_len, max_peds, 2), np.nan)
        batch_pred_gt_abs = np.full((self.batch_size, self.pred_len, max_peds, 2), np.nan)
        batch_obs_rel = np.full((self.batch_size, self.obs_len, max_peds, 2), np.nan)
        
        for i, seq_idx in enumerate(indices_in_batch):
            seq_abs = self.sequences_abs[seq_idx]
            num_peds_in_seq = seq_abs.shape[1]
            
            obs_abs = seq_abs[:self.obs_len]
            pred_abs = seq_abs[self.obs_len:]
            
            # 相対座標（速度）を計算
            obs_rel = np.zeros_like(obs_abs)
            obs_rel[1:, :, :] = obs_abs[1:, :, :] - obs_abs[:-1, :, :]
            
            batch_obs_abs[i, :, :num_peds_in_seq, :] = obs_abs
            batch_pred_gt_abs[i, :, :num_peds_in_seq, :] = pred_abs
            batch_obs_rel[i, :, :num_peds_in_seq, :] = obs_rel

        return (
            torch.from_numpy(batch_obs_rel).float(),
            torch.from_numpy(batch_obs_abs).float(),
            torch.from_numpy(batch_pred_gt_abs).float(),
        )

    def reset_batch_pointer(self):
        self.indices = np.random.permutation(self.num_sequences)
        self.pointer = 0
        
    def get_num_batches(self):
        return self.num_batches
