import numpy as np
import torch
import itertools

def getGridMask(frame, dimensions, num_person, neighborhood_size, grid_size, is_occupancy=False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 2 matrix with each row being [x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people exist in given frame
    is_occupancy: A flag using for calculation of occupancy map
    '''
    mnp = num_person

    width, height = dimensions[0], dimensions[1]
    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size**2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size**2))
    
    # Tensorをnumpyに変換
    if torch.is_tensor(frame):
        frame_np = frame.cpu().numpy()
    else:
        frame_np = frame

    width_bound = (neighborhood_size/(width*1.0))*2
    height_bound = (neighborhood_size/(height*1.0))*2

    # 全ての2-permutationsをチェック
    list_indices = list(range(0, mnp))
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
        current_x = frame_np[real_frame_index, 0]
        current_y = frame_np[real_frame_index, 1]

        width_low = current_x - width_bound/2
        width_high = current_x + width_bound/2
        height_low = current_y - height_bound/2
        height_high = current_y + height_bound/2

        other_x = frame_np[other_real_frame_index, 0]
        other_y = frame_np[other_real_frame_index, 1]
        
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
            # Ped not in surrounding, so binary mask should be zero
            continue
            
        # If in surrounding, calculate the grid cell
        cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
        cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
            continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y*grid_size] = 1
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1

    return frame_mask

def getSequenceGridMask(sequence, dimensions, pedlist_seq, neighborhood_size, grid_size, using_cuda, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A tensor of shape SL x MNP x 2
    dimensions : This will be a list [width, height]
    pedlist_seq : List of pedestrians in each frame
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of occupancy map
    '''
    sl = len(sequence) if isinstance(sequence, list) else sequence.shape[0]
    sequence_mask = []

    for i in range(sl):
        if isinstance(sequence, list):
            frame = sequence[i]
        else:
            frame = sequence[i]
            
        if isinstance(pedlist_seq, list) and i < len(pedlist_seq):
            num_peds = len(pedlist_seq[i]) if hasattr(pedlist_seq[i], '__len__') else pedlist_seq[i]
        else:
            num_peds = frame.shape[0]
            
        mask = torch.from_numpy(
            getGridMask(frame, dimensions, num_peds, neighborhood_size, grid_size, is_occupancy)
        ).float()
        
        if using_cuda and torch.cuda.is_available():
            mask = mask.cuda()
            
        sequence_mask.append(mask)

    return sequence_mask

def getSequenceGridMaskSimple(sequence, neighborhood_size, grid_size):
    '''
    Simplified version for train.py
    Get the grid masks for a single frame
    params:
    sequence : A tensor of shape (num_peds, 2)
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    num_peds = sequence.shape[0]
    frame_mask = torch.zeros((num_peds, num_peds, grid_size * grid_size))
    
    for i in range(num_peds):
        for j in range(num_peds):
            if i == j:
                continue
                
            # 相対位置を計算
            rel_pos = sequence[j] - sequence[i]
            
            # グリッドセルを計算
            cell_x = int((rel_pos[0] + neighborhood_size) / (2 * neighborhood_size / grid_size))
            cell_y = int((rel_pos[1] + neighborhood_size) / (2 * neighborhood_size / grid_size))
            
            # 境界チェック
            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                cell_idx = cell_y * grid_size + cell_x
                frame_mask[i, j, cell_idx] = 1
                
    return frame_mask
