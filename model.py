import torch
import torch.nn as nn
import numpy as np

class SocialModel(nn.Module):
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda if hasattr(args, 'use_cuda') else torch.cuda.is_available()

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.dropout_prob = args.dropout if hasattr(args, 'dropout') else 0.0
        self.seq_length = args.seq_length if hasattr(args, 'seq_length') else args.obs_seq_len + args.pred_seq_len
        self.gru = args.gru if hasattr(args, 'gru') else False

        # The LSTM/GRU cell
        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)
        else:
            self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks (numNodes, numNodes, grid_size*grid_size)
        hidden_states : Hidden states of all peds (numNodes, rnn_size)
        '''
        # Number of peds
        numNodes = grid.size(0)
        
        # Construct the social tensor
        social_tensor = torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size)
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            # grid[node]: (numNodes, grid_size*grid_size)
            # hidden_states: (numNodes, rnn_size)
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)
        
        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor

    def forward(self, input_data, grids, hidden_states, cell_states, PedsList, look_up):
        '''
        Forward pass for the model
        params:
        input_data: Input positions (seq_len, numNodes, 2)
        grids: Grid masks for each frame
        hidden_states: Hidden states of the peds (numNodes, rnn_size)
        cell_states: Cell states of the peds (numNodes, rnn_size)
        PedsList: id of peds in each frame for this sequence
        look_up: lookup table for ped ids

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states: Updated hidden states
        cell_states: Updated cell states
        '''
        
        # Get sequence length and number of nodes
        if input_data.dim() == 3:
            seq_len = input_data.size(0)
            numNodes = input_data.size(1)
        else:
            seq_len = 1
            numNodes = input_data.size(0)
            input_data = input_data.unsqueeze(0)
        
        # Construct the output variable
        outputs = torch.zeros(seq_len, numNodes, self.output_size)
        if self.use_cuda:
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum in range(seq_len):
            # Current frame data
            if input_data.dim() == 3:
                frame = input_data[framenum]
            else:
                frame = input_data
            
            # Get nodes in current frame
            if isinstance(PedsList, list) and framenum < len(PedsList):
                nodeIDs = PedsList[framenum]
                if torch.is_tensor(nodeIDs):
                    nodeIDs = nodeIDs.cpu().numpy().tolist()
                elif isinstance(nodeIDs, np.ndarray):
                    nodeIDs = nodeIDs.tolist()
                nodeIDs = [int(x) for x in nodeIDs if x < numNodes]
            else:
                # If PedsList not provided, use all nodes
                nodeIDs = list(range(numNodes))
            
            if len(nodeIDs) == 0:
                continue
            
            # Get list of node indices
            if isinstance(look_up, dict):
                list_of_nodes = [look_up.get(x, x) for x in nodeIDs]
            else:
                list_of_nodes = nodeIDs
            
            # Filter valid nodes
            list_of_nodes = [x for x in list_of_nodes if x < numNodes]
            
            if len(list_of_nodes) == 0:
                continue
                
            # Create index tensor
            corr_index = torch.LongTensor(list_of_nodes)
            if self.use_cuda:
                corr_index = corr_index.cuda()
            
            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes, :]
            
            # Get the corresponding grid mask
            if isinstance(grids, list) and framenum < len(grids):
                grid_current = grids[framenum]
                # Ensure grid has correct shape
                if grid_current.size(0) != len(list_of_nodes):
                    # Create a subset of the grid for current nodes
                    grid_current = grid_current[list_of_nodes, :, :][:, list_of_nodes, :]
            else:
                # Create dummy grid if not provided
                grid_current = torch.zeros(len(list_of_nodes), len(list_of_nodes), self.grid_size*self.grid_size)
                if self.use_cuda:
                    grid_current = grid_current.cuda()
            
            # Get the corresponding hidden and cell states
            hidden_states_current = hidden_states[list_of_nodes, :]
            
            if not self.gru:
                cell_states_current = cell_states[list_of_nodes, :]
            
            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
            
            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            
            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            
            # One-step of the LSTM/GRU
            if not self.gru:
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
                # Update cell states
                for i, node_id in enumerate(list_of_nodes):
                    cell_states[node_id] = c_nodes[i]
            else:
                h_nodes = self.cell(concat_embedded, hidden_states_current)
            
            # Compute the output
            for i, node_id in enumerate(list_of_nodes):
                outputs[framenum, node_id, :] = self.output_layer(h_nodes[i:i+1])
                # Update hidden states
                hidden_states[node_id] = h_nodes[i]
        
        return outputs, hidden_states, cell_states
