import os
import pickle
import numpy as np
import torch

class DataLoader:
    def __init__(self, data_dir, batch_size=50, seq_length=8, pred_length=12, forcePreProcess=False):
        """
        Social-STGCNNのデータセット形式に合わせたDataLoader
        Args:
            data_dir (str): データセット（例: 'eth', 'hotel'）を含む親ディレクトリのパス
            batch_size (int): バッチサイズ
            seq_length (int): 観測フレーム数 (STGCNNでは通常8)
            pred_length (int): 予測フレーム数 (STGCNNでは通常12)
            forcePreProcess (bool): 前処理を強制的に再実行するか
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.total_length = seq_length + pred_length

        # データセット名を取得 (例: 'eth', 'hotel')
        self.dataset_name = os.path.basename(data_dir)
        
        # データファイルのパスを定義
        data_file = os.path.join(self.data_dir, f"{self.dataset_name}.ndjson")

        # 前処理済みファイルのパスを定義
        processed_data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        if not os.path.exists(processed_data_file) or forcePreProcess:
            print(f"Creating pre-processed data for {self.dataset_name}...")
            self.frame_preprocess(data_file, processed_data_file)

        self.load_preprocessed(processed_data_file)
        self.reset_batch_pointer()

    def frame_preprocess(self, data_file, out_file):
        """
        生の軌跡データ（.ndjson or .txt）を前処理する
        """
        all_peds = {}
        with open(data_file, 'r') as f:
            for line in f:
                # Social-STGCNNはタブ区切り
                data = line.strip().split('\t')
                frame_id, ped_id, x, y = [float(v) for v in data]
                
                if ped_id not in all_peds:
                    all_peds[ped_id] = []
                all_peds[ped_id].append([frame_id, x, y])
        
        # 軌跡をNumpy配列に変換し、フレームIDでソート
        trajectories = []
        for ped_id, traj in all_peds.items():
            traj = np.array(traj)
            traj = traj[np.argsort(traj[:, 0])] # フレームIDでソート
            trajectories.append(traj)

        with open(out_file, "wb") as f:
            pickle.dump(trajectories, f)

    def load_preprocessed(self, data_file):
        """
        前処理済みデータをロードし、シーケンスを作成
        """
        with open(data_file, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.sequences = []
        for traj in self.trajectories:
            # total_length分の長さを持つシーケンスを切り出す
            for i in range(len(traj) - self.total_length + 1):
                start = i
                end = i + self.total_length
                seq = traj[start:end, 1:3] # x,y座標のみ使用
                self.sequences.append(seq)
        
        self.num_sequences = len(self.sequences)
        self.num_batches = int(self.num_sequences / self.batch_size)
        print(f"Loaded {self.num_sequences} sequences from {self.dataset_name}, creating {self.num_batches} batches.")

    def reset_batch_pointer(self):
        """
        バッチポインタをリセット
        """
        # シーケンスをシャッフル
        np.random.shuffle(self.sequences)
        self.pointer = 0

    def next_batch(self):
        """
        次のバッチを生成
        """
        if self.pointer + self.batch_size > self.num_sequences:
            self.reset_batch_pointer()

        batch_sequences = self.sequences[self.pointer : self.pointer + self.batch_size]
        self.pointer += self.batch_size

        batch = np.array(batch_sequences)
        
        # 観測データと予測対象データに分割
        obs_traj = torch.from_numpy(batch[:, :self.seq_length, :]).float()
        pred_traj = torch.from_numpy(batch[:, self.seq_length:, :]).float()

        return obs_traj, pred_traj

    def get_num_batches(self):
        return self.num_batches

