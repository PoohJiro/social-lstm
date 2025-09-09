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
        
        hidden_states = torch.zeros(batch_size * num_peds, self.rnn_size).to(device)
        cell_states = torch.zeros(batch_size * num_peds, self.rnn_size).to(device)

        # --- Encoder ---
        for t in range(obs_len):
            frame_obs_rel = obs_traj_rel[:, t].reshape(-1, 2)
            grid_mask = grids[:, t].reshape(batch_size * num_peds, num_peds, -1)
            
            social_tensor = self.get_social_tensor(grid_mask, hidden_states)
            input_embedded = self.relu(self.input_embedding(frame_obs_rel))
            tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
            
            concat_embedded = torch.cat([input_embedded, tensor_embedded], dim=1)
            
            h, c = self.cell(concat_embedded, (hidden_states, cell_states))
            hidden_states, cell_states = h, c
        
        # --- Decoder ---
        predictions = []
        last_pos_rel = obs_traj_rel[:, -1].reshape(-1, 2)
        
        for _ in range(self.args.pred_len):
            social_tensor = self.get_social_tensor(grids[:, -1].reshape(batch_size * num_peds, num_peds, -1), hidden_states)
            input_embedded = self.relu(self.input_embedding(last_pos_rel))
            tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
            
            concat_embedded = torch.cat([input_embedded, tensor_embedded], dim=1)
            
            h, c = self.cell(concat_embedded, (hidden_states, cell_states))
            hidden_states, cell_states = h, c
            
            output = self.output_layer(hidden_states)
            predictions.append(output)
            last_pos_rel = output
            
        return torch.stack(predictions, dim=1).reshape(batch_size, self.args.pred_len, num_peds, 2)
