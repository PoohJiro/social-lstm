import torch
import torch.nn as nn

class SocialModel(nn.Module):
    """
    Social-STGCNNデータセットで動作する、シンプルなEncoder-Decoder LSTMモデル
    """
    def __init__(self, args):
        super(SocialModel, self).__init__()

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.use_gru = args.gru

        # --- Encoder ---
        # 座標(x,y)を埋め込みベクトルに変換
        self.input_embedding = nn.Linear(args.input_size, self.embedding_size)
        
        # LSTMまたはGRU層
        if self.use_gru:
            self.encoder = nn.GRU(self.embedding_size, self.rnn_size, batch_first=True)
        else:
            self.encoder = nn.LSTM(self.embedding_size, self.rnn_size, batch_first=True)

        # --- Decoder ---
        # Decoderも同じRNNセルを使用
        if self.use_gru:
            self.decoder = nn.GRU(self.embedding_size, self.rnn_size, batch_first=True)
        else:
            self.decoder = nn.LSTM(self.embedding_size, self.rnn_size, batch_first=True)
            
        # RNNの出力（隠れ状態）を座標(x,y)に変換
        self.output_layer = nn.Linear(self.rnn_size, args.output_size)
        
        # 活性化関数
        self.relu = nn.ReLU()

    def forward(self, obs_traj):
        """
        Args:
            obs_traj (torch.Tensor): 観測された軌跡 (batch_size, obs_len, 2)
        Returns:
            pred_traj (torch.Tensor): 予測された軌跡 (batch_size, pred_len, 2)
        """
        batch_size = obs_traj.size(0)

        # --- Encoding Phase ---
        # (batch, obs_len, 2) -> (batch, obs_len, embedding_size)
        embedded_obs = self.relu(self.input_embedding(obs_traj))
        
        # エンコーダーで軌跡全体を処理し、最後の隠れ状態（と記憶セル）を取得
        _, hidden_state = self.encoder(embedded_obs)
        
        # --- Decoding Phase ---
        # 予測された軌跡を格納するリスト
        predictions = []
        
        # デコーダーの最初の入力は、観測の最後の座標
        decoder_input = embedded_obs[:, -1, :].unsqueeze(1) # (batch, 1, embedding_size)

        for _ in range(self.pred_len):
            # デコーダーを1ステップ実行
            output, hidden_state = self.decoder(decoder_input, hidden_state)
            
            # 出力を座標に変換
            # (batch, 1, rnn_size) -> (batch, 1, 2)
            pred_step = self.output_layer(output)
            predictions.append(pred_step)
            
            # 次のステップの入力は、今回予測した座標を埋め込んだもの
            decoder_input = self.relu(self.input_embedding(pred_step))

        # 予測を一つのテンソルにまとめる
        # [(batch, 1, 2), (batch, 1, 2), ...] -> (batch, pred_len, 2)
        pred_traj = torch.cat(predictions, dim=1)

        return pred_traj
