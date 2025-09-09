# model.py
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SocialModel(nn.Module):
    def __init__(self, args):
        super(SocialModel, self).__init__()
        self.args = args
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        
        self.input_embedding = nn.Linear(args.input_size, self.embedding_size)
        self.tensor_embedding = nn.Linear(self.grid_size * self.grid_size * self.rnn_size, self.embedding_size)
        self.output_layer = nn.Linear(self.rnn_size, args.output_size)
        self.relu = nn.ReLU()

        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)

    def get_social_tensor(self, grid_mask, hidden_states):
        num_peds = hidden_states.shape[0]
        grid_mask_t = grid_mask.permute(0, 2, 1)
        social_tensor = torch.matmul(grid_mask_t, hidden_states)
        return social_tensor.view(num_peds, -1)

    def forward(self, obs_traj_rel, grids):
        batch_size, obs_len, num_peds, _ = obs_traj_rel.shape
        
        # バッチと歩行者の次元を統合して効率化
        hidden_states = torch.zeros(batch_size * num_peds, self.rnn_size).to(device)
        cell_states = torch.zeros(batch_size * num_peds, self.rnn_size).to(device)

        # --- Encoder ---
        for t in range(obs_len):
            # (batch, num_peds, 2) -> (batch * num_peds, 2)
            frame_obs_rel = obs_traj_rel[:, t].reshape(-1, 2)
            # (batch, num_peds, num_peds, grid*grid) -> (batch*num_peds, num_peds, grid*grid)
            grid_mask = grids[:, t].reshape(batch_size * num_peds, num_peds, -1)
            
            # NaNの歩行者（パディング）をマスク
            nan_mask = ~torch.isnan(frame_obs_rel).any(dim=1)
            
            social_tensor = self.get_social_tensor(grid_mask[nan_mask], hidden_states[nan_mask])
            input_embedded = self.relu(self.input_embedding(frame_obs_rel[nan_mask]))
            tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
            
            concat_embedded = torch.cat([input_embedded, tensor_embedded], dim=1)
            
            h, c = self.cell(concat_embedded, (hidden_states[nan_mask], cell_states[nan_mask]))
            hidden_states[nan_mask] = h
            cell_states[nan_mask] = c
        
        # --- Decoder ---
        predictions_rel = []
        last_pos_rel = obs_traj_rel[:, -1].reshape(-1, 2)
        
        for _ in range(self.args.pred_len):
            grid_mask = grids[:, -1].reshape(batch_size * num_peds, num_peds, -1)
            nan_mask = ~torch.isnan(last_pos_rel).any(dim=1)
            
            social_tensor = self.get_social_tensor(grid_mask[nan_mask], hidden_states[nan_mask])
            input_embedded = self.relu(self.input_embedding(last_pos_rel[nan_mask]))
            tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
            
            concat_embedded = torch.cat([input_embedded, tensor_embedded], dim=1)
            
            h, c = self.cell(concat_embedded, (hidden_states[nan_mask], cell_states[nan_mask]))
            hidden_states[nan_mask] = h
            cell_states[nan_mask] = c
            
            output = self.output_layer(hidden_states)
            predictions_rel.append(output)
            last_pos_rel = output
            
        # (pred_len, batch*peds, 2) -> (batch, peds, pred_len, 2) -> (batch, pred_len, peds, 2)
        pred_traj_fake_rel = torch.stack(predictions_rel).permute(1, 0, 2).reshape(batch_size, num_peds, self.args.pred_len, 2).permute(0, 2, 1, 3)
        return pred_traj_fake_rel
