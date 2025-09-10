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
        
        # Initialize hidden states per batch item
        hidden_states = torch.zeros(batch_size, num_peds, self.rnn_size).to(device)
        cell_states = torch.zeros(batch_size, num_peds, self.rnn_size).to(device)

        # Encoder
        for t in range(obs_len):
            for b in range(batch_size):
                frame_obs_rel = obs_traj_rel[b, t] # (num_peds, 2)
                grid_mask = grids[b, t]            # (num_peds, num_peds, 16)
                h_b = hidden_states[b]             # (num_peds, rnn_size)
                c_b = cell_states[b]               # (num_peds, rnn_size)
                
                nan_mask = ~torch.isnan(frame_obs_rel).any(dim=1)
                if not nan_mask.any(): continue

                # Filter all tensors to only valid pedestrians for this batch item
                grid_valid = grid_mask[nan_mask, :, :][:, nan_mask]
                h_valid = h_b[nan_mask]
                c_valid = c_b[nan_mask]
                frame_obs_valid = frame_obs_rel[nan_mask]
                
                social_tensor = self.get_social_tensor(grid_valid, h_valid)
                input_embedded = self.relu(self.input_embedding(frame_obs_valid))
                tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
                concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
                
                h_next, c_next = self.cell(concat_embedded, (h_valid, c_valid))
                
                # Update the original hidden state tensor
                hidden_states[b, nan_mask] = h_next
                cell_states[b, nan_mask] = c_next
        
        # Decoder
        predictions = []
        last_pos_rel = obs_traj_rel[:, -1] # (batch, num_peds, 2)
        
        for _ in range(self.args.pred_len):
            batch_outputs = torch.full((batch_size, num_peds, self.args.output_size), float('nan')).to(device)
            next_pos_batch = torch.full_like(last_pos_rel, float('nan'))

            for b in range(batch_size):
                last_pos_rel_b = last_pos_rel[b]
                grid_mask = grids[b, -1] # Use last observed grid for all pred steps
                h_b = hidden_states[b]
                c_b = cell_states[b]
                
                nan_mask = ~torch.isnan(last_pos_rel_b).any(dim=1)
                if not nan_mask.any(): continue
                
                grid_valid = grid_mask[nan_mask, :, :][:, nan_mask]
                h_valid = h_b[nan_mask]
                c_valid = c_b[nan_mask]
                last_pos_valid = last_pos_rel_b[nan_mask]

                social_tensor = self.get_social_tensor(grid_valid, h_valid)
                input_embedded = self.relu(self.input_embedding(last_pos_valid))
                tensor_embedded = self.relu(self.tensor_embedding(social_tensor))
                concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

                h_next, c_next = self.cell(concat_embedded, (h_valid, c_valid))
                
                output_params = self.output_layer(h_next)
                
                # Update states and store outputs
                hidden_states[b, nan_mask] = h_next
                cell_states[b, nan_mask] = c_next
                batch_outputs[b, nan_mask, :] = output_params
                next_pos_batch[b, nan_mask, :] = output_params[:, :2]

            predictions.append(batch_outputs.unsqueeze(1))
            last_pos_rel = next_pos_batch

        pred_params = torch.cat(predictions, dim=1)
        return pred_params
