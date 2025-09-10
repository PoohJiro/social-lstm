import torch

def get_grid_mask_vectorized(sequence, neighborhood_size, grid_size, use_cuda):
    """
    Computes the grid mask for an entire sequence of pedestrian positions using vectorized operations.
    
    params:
    sequence : A tensor of shape (seq_len, num_peds, 2).
    neighborhood_size : Scalar value.
    grid_size : Scalar value.
    use_cuda: Boolean indicating if CUDA is used.
    """
    seq_len, num_peds, _ = sequence.shape
    device = sequence.device
    
    if num_peds < 2:
        return torch.zeros((seq_len, num_peds, num_peds, grid_size * grid_size), device=device)

    # Calculate all-pairs relative positions for the entire sequence.
    # [seq, num_peds, 1, 2] - [seq, 1, num_peds, 2] -> [seq, num_peds, num_peds, 2]
    rel_pos = sequence.unsqueeze(2) - sequence.unsqueeze(1)
    
    div_factor = (2.0 * neighborhood_size / grid_size)
    cell_x = torch.floor((rel_pos[..., 0] + neighborhood_size) / div_factor).long()
    cell_y = torch.floor((rel_pos[..., 1] + neighborhood_size) / div_factor).long()
    
    # Boundary checks
    in_bounds = (cell_x >= 0) & (cell_x < grid_size) & (cell_y >= 0) & (cell_y < grid_size)
    
    cell_idx = cell_y * grid_size + cell_x
    
    # Create the final mask
    mask = torch.zeros((seq_len, num_peds, num_peds, grid_size * grid_size), device=device)
    
    # Use the in_bounds mask to create indices for scatter
    in_bounds_indices = in_bounds.nonzero(as_tuple=True)
    valid_cell_idx = cell_idx[in_bounds_indices]

    # Use advanced indexing to place '1's in the correct grid cells
    mask[in_bounds_indices[0], in_bounds_indices[1], in_bounds_indices[2], valid_cell_idx] = 1
    
    # The diagonal should be zero
    # Reshape to easily access diagonals for all frames at once
    mask.view(seq_len, num_peds, num_peds, -1).diagonal(dim1=1, dim2=2).zero_()
    
    return mask
