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

    def get_social_tensor(self, grid, hidden_states):
        """
        Computes the social tensor using a for loop for clarity and correctness,
        matching the original paper's logic.
        """
        num_peds = hidden_states.shape[0]
        # Return zero tensor if no peds are present
        if num_peds == 0:
            return torch.zeros(0, self.grid_size * self.grid_size * self.rnn_size).to(device)

        social_tensor = torch.zeros(num_peds, self.grid_size*self.grid_size, self.rnn_size).to(device)
        
        # For each pedestrian
        for node in range(num_peds):
            # Compute the social tensor
            # grid[node] shape: (num_peds, grid_size*grid_size)
            # torch.t(grid[node]) shape: (grid_size*grid_size, num_peds)
            # hidden_states shape: (num_peds, rnn_size)
            # Result is (grid_size*grid_size, rnn_size)
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)
            
        return social_tensor.view(num_peds, self.grid_size * self.grid_size * self.rnn_size)
        
    def forward(self, obs_traj_rel, grids):
        batch_size, obs_len, num_peds, _ = obs_traj_rel.shape
        
        hidden_states = torch.zeros(batch_size * num_peds, self.rnn_size).to(device)
        cell_states = torch.zeros(batch_size * num_peds, self.rnn_size).to(device)

        # Encoder
        for t in range(obs_len):
            frame_obs_rel = obs_traj_rel[:, t].reshape(-1, 2)
            grid_mask = grids[:, t].reshape(batch_size * num_peds, num_peds, -1)
            nan_mask = ~torch.isnan(frame_obs_rel).any(dim=1)
            if not nan_mask.any(): continue
            
            social_tensor = self.get_social_tensor(grid_mask[nan_mask], hidden_states[nan_mask])
            input_embedded = self.relu(self.input_embedding(frame_obs_rel[nan_mask]))
            tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            h, c = self.cell(concat_embedded, (hidden_states[nan_mask], cell_states[nan_mask]))
            hidden_states[nan_mask], cell_states[nan_mask] = h, c
        
        # Decoder
        predictions = []
        last_pos_rel = obs_traj_rel[:, -1].reshape(-1, 2)
        for _ in range(self.args.pred_len):
            nan_mask = ~torch.isnan(last_pos_rel).any(dim=1)
            if not nan_mask.any():
                predictions.append(torch.full((batch_size * num_peds, self.args.output_size), float('nan')).to(device))
                last_pos_rel = torch.zeros_like(last_pos_rel)
                continue
                
            grid_mask = grids[:, -1].reshape(batch_size * num_peds, num_peds, -1)
            social_tensor = self.get_social_tensor(grid_mask[nan_mask], hidden_states[nan_mask])
            input_embedded = self.relu(self.input_embedding(last_pos_rel[nan_mask]))
            tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            h, c = self.cell(concat_embedded, (hidden_states[nan_mask], cell_states[nan_mask]))
            hidden_states[nan_mask], cell_states[nan_mask] = h, c
            
            output_params = self.output_layer(h)
            
            next_pos = torch.full_like(last_pos_rel, float('nan'))
            next_pos[nan_mask] = output_params[:, :2]
            
            full_output = torch.full((batch_size * num_peds, self.args.output_size), float('nan')).to(device)
            full_output[nan_mask] = output_params
            predictions.append(full_output)
            last_pos_rel = next_pos
            
        pred_params = torch.stack(predictions).permute(1, 0, 2).reshape(batch_size, num_peds, self.args.pred_len, self.args.output_size).permute(0, 2, 1, 3)
        return pred_params
