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
        
        # 歩行者IDごとの軌跡に変換
        num_peds = int(np.max(data[:, 1]))
        traj_data = np.zeros((len(frames), num_peds + 1, 4)) # frame, ped_id, x, y
        traj_data[:] = np.nan

        for idx, frame in enumerate(frame_data):
            for ped in frame:
                traj_data[idx, int(ped[1])] = ped
        
        # pickleで保存
        with open(out_file, "wb") as f:
            pickle.dump(traj_data, f)

    def load_preprocessed(self, data_file):
        with open(data_file, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.peds_in_seq = []
        self.sequences_abs = [] # 絶対座標
        self.sequences_rel = [] # 相対座標 (速度)

        # self.seq_len の長さを持つシーケンスを切り出す
        for i in range(self.raw_data.shape[0] - self.seq_len + 1):
            seq_abs = self.raw_data[i:i+self.seq_len, :, 2:4].copy()
            # このシーケンスに登場する歩行者（NaNでない軌跡）のIDを取得
            peds_present = ~np.isnan(seq_abs).all(axis=(0, 2))
            
            if not np.any(peds_present):
                continue
            
            # 歩行者がいない列を削除
            seq_abs = seq_abs[:, peds_present, :]
            
            # 相対座標を計算
            seq_rel = np.zeros_like(seq_abs)
            seq_rel[1:, :, :] = seq_abs[1:, :, :] - seq_abs[:-1, :, :]
            
            self.sequences_abs.append(seq_abs)
            self.sequences_rel.append(seq_rel)

        self.num_sequences = len(self.sequences_abs)
        self.num_batches = int(self.num_sequences / self.batch_size)
        print(f"Loaded {self.num_sequences} sequences, creating {self.num_batches} batches.")

    def next_batch(self):
        if self.pointer + self.batch_size > self.num_sequences:
            self.reset_batch_pointer()

        start = self.pointer
        end = self.pointer + self.batch_size
        self.pointer += self.batch_size
        
        # バッチ内の最大歩行者数を取得
        max_peds = 0
        for i in range(start, end):
            max_peds = max(max_peds, self.sequences_abs[i].shape[1])
        
        # バッチデータを格納するテンソルを初期化
        batch_obs_abs = np.zeros((self.batch_size, self.obs_len, max_peds, 2))
        batch_obs_rel = np.zeros((self.batch_size, self.obs_len, max_peds, 2))
        batch_pred_gt_abs = np.zeros((self.batch_size, self.pred_len, max_peds, 2))
        batch_pred_gt_rel = np.zeros((self.batch_size, self.pred_len, max_peds, 2))
        
        # バッチ内の各シーケンスをパディングして格納
        for i, idx in enumerate(range(start, end)):
            seq_abs = self.sequences_abs[idx]
            seq_rel = self.sequences_rel[idx]
            num_peds = seq_abs.shape[1]
            
            batch_obs_abs[i, :, :num_peds, :] = seq_abs[:self.obs_len]
            batch_obs_rel[i, :, :num_peds, :] = seq_rel[:self.obs_len]
            batch_pred_gt_abs[i, :, :num_peds, :] = seq_abs[self.obs_len:]
            batch_pred_gt_rel[i, :, :num_peds, :] = seq_rel[self.obs_len:]

        return (
            torch.from_numpy(batch_obs_rel).float(),
            torch.from_numpy(batch_pred_gt_rel).float(),
            torch.from_numpy(batch_obs_abs).float(),
            torch.from_numpy(batch_pred_gt_abs).float(),
        )

    def reset_batch_pointer(self):
        # シーケンスのインデックスをシャッフル
        self.indices = np.random.permutation(self.num_sequences)
        self.pointer = 0
        
    def get_num_batches(self):
        return self.num_batches

