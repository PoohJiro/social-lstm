import torch

def getSequenceGridMask(sequence, neighborhood_size, grid_size, use_cuda):
    """
    Computes the grid mask for a single frame of pedestrian positions using vectorized operations.
    
    params:
    sequence : A tensor of shape (num_peds, 2) representing ped positions in a frame.
    neighborhood_size : Scalar value representing the size of the neighborhood.
    grid_size : Scalar value representing the size of the grid discretization.
    use_cuda: Boolean indicating if CUDA is used (for device placement).
    """
    num_peds = sequence.size(0)
    if num_peds < 2:
        # If there's only one or zero pedestrians, no social interaction exists.
        return torch.zeros((num_peds, num_peds, grid_size * grid_size), device=sequence.device)

    # --- Vectorized Grid Mask Calculation ---
    
    # 1. Calculate all-pairs relative positions in one go.
    # [num_peds, 1, 2] - [1, num_peds, 2] -> [num_peds, num_peds, 2]
    rel_pos = sequence.unsqueeze(1) - sequence.unsqueeze(0)
    
    # 2. Calculate grid cell indices for all pairs at once.
    # The division factor, ensuring floating point division.
    div_factor = (2.0 * neighborhood_size / grid_size)
    cell_x = torch.floor((rel_pos[:, :, 0] + neighborhood_size) / div_factor).long()
    cell_y = torch.floor((rel_pos[:, :, 1] + neighborhood_size) / div_factor).long()
    
    # 3. Perform boundary checks for all pairs at once.
    in_bounds = (cell_x >= 0) & (cell_x < grid_size) & (cell_y >= 0) & (cell_y < grid_size)
    
    # 4. Calculate the final linear grid cell index.
    cell_idx = cell_y * grid_size + cell_x
    
    # 5. Create the final mask.
    mask = torch.zeros((num_peds, num_peds, grid_size * grid_size), device=sequence.device)
    
    # Use the boundary check mask to select valid indices.
    valid_cell_idx = cell_idx[in_bounds]
    
    # To use scatter_, we need to get the indices of the 'True' values from `in_bounds`.
    # `nonzero()` gives us the row and column indices for each valid pair.
    row_indices, col_indices = in_bounds.nonzero(as_tuple=True)
    
    # We need to reshape valid_cell_idx to be compatible for scatter.
    # It should be of the same shape as the index tensor (row_indices, col_indices).
    # We add a new dimension to use it as the index for the last dimension of the mask.
    valid_cell_idx = valid_cell_idx.unsqueeze(1)

    # Prepare a tensor of ones to scatter into the mask
    source = torch.ones_like(valid_cell_idx, dtype=mask.dtype)
    
    # Use scatter_ to place '1's in the correct grid cells for each valid pair.
    # The indices for scatter need to be constructed carefully.
    # Here, we create a tensor of indices for the first two dimensions.
    idx_tensor = torch.stack([row_indices, col_indices], dim=0).unsqueeze(2)
    # We need to expand it to match the dimensions for scattering.
    # This is a bit complex, let's use a simpler approach with a direct mask update.

    # A more straightforward approach than scatter for this case:
    # Flatten the first two dimensions to easily update the mask
    flat_mask = mask.view(-1, grid_size * grid_size)
    
    # Create flat indices for updating
    # (row * num_peds + col) gives the flat index for the (row, col) pair
    flat_pair_indices = row_indices * num_peds + col_indices
    
    # Update the mask using `index_put_` which is often easier to reason about
    # for this kind of assignment.
    # We update `flat_mask` at `flat_pair_indices` and `valid_cell_idx`.
    # `index_put_` takes a tuple of index tensors.
    flat_mask.index_put_((flat_pair_indices, valid_cell_idx.squeeze()), source.squeeze())
    
    # Reshape the mask back to its original 3D shape
    mask = flat_mask.view(num_peds, num_peds, -1)
    
    # The diagonal should be zero (a person is not in their own neighborhood grid).
    mask.diagonal().zero_()
    
    return mask
