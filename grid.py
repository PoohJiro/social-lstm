import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_grid_mask(frame_data, neighborhood_size, grid_size):
    """
    各歩行者の近傍グリッドマスクを計算する。NaN（パディング）を考慮する。
    """
    num_peds = frame_data.shape[0]
    
    # NaN（パディング）された歩行者を除外して計算
    nan_mask = ~torch.isnan(frame_data).any(dim=-1)
    frame_data_nonan = frame_data[nan_mask]
    num_peds_nonan = frame_data_nonan.shape[0]

    # 有効な歩行者がいなければ、ゼロのグリッドを返す
    if num_peds_nonan == 0:
        return torch.zeros(num_peds, num_peds, grid_size * grid_size, device=device)

    cell_size = neighborhood_size / grid_size
    grid_centers_x = torch.arange(-neighborhood_size / 2 + cell_size / 2, neighborhood_size / 2, cell_size, device=device)
    grid_centers_y = torch.arange(-neighborhood_size / 2 + cell_size / 2, neighborhood_size / 2, cell_size, device=device)
    grid_centers = torch.stack(torch.meshgrid(grid_centers_x, grid_centers_y, indexing='ij'), dim=-1).reshape(-1, 2)
    
    ped_grid_centers = frame_data_nonan.unsqueeze(1) + grid_centers.unsqueeze(0)
    dist = ped_grid_centers.unsqueeze(1) - frame_data_nonan.unsqueeze(0).unsqueeze(2)
    
    mask_nonan = (torch.abs(dist[..., 0]) < cell_size / 2) & (torch.abs(dist[..., 1]) < cell_size / 2)
    
    # 元の歩行者数に合わせたフルサイズのマスクを作成
    full_mask = torch.zeros(num_peds, num_peds, grid_size * grid_size, device=device)
    
    # 計算したマスクを、有効な歩行者のインデックスに対応する位置に挿入
    valid_indices = torch.where(nan_mask)[0]
    row_indices, col_indices = torch.meshgrid(valid_indices, valid_indices, indexing='ij')
    
    # ★★★ バグ修正箇所 ★★★
    # 不要な .permute() を削除し、.float()で型を合わせる
    full_mask[row_indices, col_indices, :] = mask_nonan.float()
    
    return full_mask
