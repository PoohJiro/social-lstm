import torch
import numpy as np

def getSequenceGridMask(sequence, neighborhood_size, grid_size, use_cuda):
    """
    Computes the grid mask for a single frame of pedestrian positions.
    This is the sole function in this file to avoid import conflicts.
    
    params:
    sequence : A tensor of shape (num_peds, 2) representing ped positions in a frame.
    neighborhood_size : Scalar value representing the size of the neighborhood.
    grid_size : Scalar value representing the size of the grid discretization.
    use_cuda: Boolean indicating if CUDA is used (for device placement).
    """
    num_peds = sequence.size(0)
    device = sequence.device
    frame_mask = torch.zeros((num_peds, num_peds, grid_size * grid_size), device=device)
    
    for i in range(num_peds):
        for j in range(num_peds):
            if i == j:
                continue
                
            # Calculate relative position
            rel_pos = sequence[j] - sequence[i]
            
            # Calculate grid cell index
            # Ensure division is not integer division
            cell_x = int((rel_pos[0] + neighborhood_size) / (2.0 * neighborhood_size / grid_size))
            cell_y = int((rel_pos[1] + neighborhood_size) / (2.0 * neighborhood_size / grid_size))
            
            # Boundary check
            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                cell_idx = cell_y * grid_size + cell_x
                frame_mask[i, j, int(cell_idx)] = 1
                
    return frame_mask
