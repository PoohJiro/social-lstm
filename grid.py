# grid.py
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_grid_mask(frame_data, neighborhood_size, grid_size):
    num_peds = frame_data.shape[0]
    cell_size = neighborhood_size / grid_size
    grid_centers_x = torch.arange(-neighborhood_size / 2 + cell_size / 2, neighborhood_size / 2, cell_size, device=device)
    grid_centers_y = torch.arange(-neighborhood_size / 2 + cell_size / 2, neighborhood_size / 2, cell_size, device=device)
    grid_centers = torch.stack(torch.meshgrid(grid_centers_x, grid_centers_y, indexing='ij'), dim=-1).reshape(-1, 2)
    ped_grid_centers = frame_data.unsqueeze(1) + grid_centers.unsqueeze(0)
    dist = ped_grid_centers.unsqueeze(1) - frame_data.unsqueeze(1).unsqueeze(2)
    mask = (torch.abs(dist[..., 0]) < cell_size / 2) & (torch.abs(dist[..., 1]) < cell_size / 2)
    grid_mask = mask.sum(dim=-1).float().permute(0, 2, 1)
    return grid_mask.permute(0, 2, 1)
