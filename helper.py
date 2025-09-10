import numpy as np
import torch
import os
import shutil
from os import walk
import math

# モデルのインポート（必要に応じて）
try:
    from model import SocialModel
    from olstm_model import OLSTMModel
    from vlstm_model import VLSTMModel
except ImportError:
    # モデルが見つからない場合は無視
    pass

# one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)

# (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)
def get_method_name(index):
    # return method name given index
    return {
        1: 'SOCIALLSTM',
        2: 'OBSTACLELSTM',
        3: 'VANILLALSTM'
    }.get(index, 'SOCIALLSTM')

def get_model(index, arguments, infer=False):
    # return a model given index and arguments
    if index == 1:
        return SocialModel(arguments, infer)
    elif index == 2:
        return OLSTMModel(arguments, infer)
    elif index == 3:
        return VLSTMModel(arguments, infer)
    else:
        return SocialModel(arguments, infer)

def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    # outputsが5次元の場合（ガウス分布パラメータ）
    if outputs.size(-1) >= 5:
        mux = outputs[:, :, 0]
        muy = outputs[:, :, 1]
        sx = outputs[:, :, 2]
        sy = outputs[:, :, 3]
        corr = outputs[:, :, 4]
        
        sx = torch.exp(sx)
        sy = torch.exp(sy)
        corr = torch.tanh(corr)
    else:
        # outputsが2次元の場合（座標のみ）
        mux = outputs[:, :, 0]
        muy = outputs[:, :, 1]
        sx = torch.ones_like(mux)
        sy = torch.ones_like(muy)
        corr = torch.zeros_like(mux)
    
    return mux, muy, sx, sy, corr

def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters
    ==========
    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes or numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation
    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======
    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    # 次元を調整
    if mux.dim() == 2:
        o_mux = mux[0, :]
        o_muy = muy[0, :]
        o_sx = sx[0, :]
        o_sy = sy[0, :]
        o_corr = corr[0, :]
    else:
        o_mux = mux
        o_muy = muy
        o_sx = sx
        o_sy = sy
        o_corr = corr

    numNodes = o_mux.size(0)
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    
    # look_upが辞書の場合の処理
    if isinstance(look_up, dict):
        converted_node_present = [look_up.get(node, node) for node in nodesPresent]
    else:
        converted_node_present = nodesPresent
    
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
            
        mean = [o_mux[node].item(), o_muy[node].item()]
        sx_val = o_sx[node].item()
        sy_val = o_sy[node].item()
        corr_val = o_corr[node].item()
        
        cov = [[sx_val * sx_val, corr_val * sx_val * sy_val], 
               [corr_val * sx_val * sy_val, sy_val * sy_val]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        
        # 共分散行列が正定値になるように小さな値を追加
        cov = cov + np.eye(2) * 1e-6
        
        try:
            next_values = np.random.multivariate_normal(mean, cov, 1)
            next_x[node] = next_values[0][0]
            next_y[node] = next_values[0][1]
        except:
            # エラーが発生した場合は平均値を使用
            next_x[node] = mean[0]
            next_y[node] = mean[1]

    return next_x, next_y

def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========
    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes
    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes
    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======
    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size(0)
    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()

    for tstep in range(pred_length):
        counter = 0
        
        # assumedNodesPresentが提供されていない場合
        if assumedNodesPresent is None or tstep >= len(assumedNodesPresent):
            # 全ノードを使用
            for nodeID in range(ret_nodes.size(1)):
                pred_pos = ret_nodes[tstep, nodeID, :]
                true_pos = nodes[tstep, nodeID, :]
                error[tstep] += torch.norm(pred_pos - true_pos, p=2)
                counter += 1
        else:
            for nodeID in assumedNodesPresent[tstep]:
                nodeID = int(nodeID)
                
                if trueNodesPresent is not None and tstep < len(trueNodesPresent):
                    if nodeID not in trueNodesPresent[tstep]:
                        continue
                
                if isinstance(look_up, dict):
                    nodeID = look_up.get(nodeID, nodeID)
                
                if nodeID >= ret_nodes.size(1):
                    continue
                    
                pred_pos = ret_nodes[tstep, nodeID, :]
                true_pos = nodes[tstep, nodeID, :]
                error[tstep] += torch.norm(pred_pos - true_pos, p=2)
                counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error)

def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    '''
    Parameters
    ==========
    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes
    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes
    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======
    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size(0)
    error = 0
    counter = 0

    # Last time-step
    tstep = pred_length - 1
    
    # assumedNodesPresentが提供されていない場合
    if assumedNodesPresent is None or tstep >= len(assumedNodesPresent):
        # 全ノードを使用
        for nodeID in range(ret_nodes.size(1)):
            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]
            error += torch.norm(pred_pos - true_pos, p=2)
            counter += 1
    else:
        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)
            
            if trueNodesPresent is not None and tstep < len(trueNodesPresent):
                if nodeID not in trueNodesPresent[tstep]:
                    continue
            
            if isinstance(look_up, dict):
                nodeID = look_up.get(nodeID, nodeID)
            
            if nodeID >= ret_nodes.size(1):
                continue
                
            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]
            error += torch.norm(pred_pos - true_pos, p=2)
            counter += 1
        
    if counter != 0:
        error = error / counter
            
    return error

def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
    '''
    params:
    outputs : predicted locations
    targets : true locations
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index
    '''
    seq_length = outputs.size(0)
    
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    # 数値安定性のため小さな値を追加
    sx = sx + 1e-6
    sy = sy + 1e-6
    sxsy = sxsy + 1e-6

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # 数値安定性のため
    negRho = torch.clamp(negRho, min=1e-6)

    # Numerator
    result = torch.exp(-z/(2*negRho))
    
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0
    counter = 0

    for framenum in range(seq_length):
        if nodesPresent is not None and framenum < len(nodesPresent):
            nodeIDs = nodesPresent[framenum]
            if torch.is_tensor(nodeIDs):
                nodeIDs = nodeIDs.cpu().numpy().tolist()
            nodeIDs = [int(nodeID) for nodeID in nodeIDs]
        else:
            # nodesPresentが提供されていない場合は全ノードを使用
            nodeIDs = list(range(outputs.size(1)))

        for nodeID in nodeIDs:
            if isinstance(look_up, dict):
                nodeID = look_up.get(nodeID, nodeID)
            
            if nodeID >= outputs.size(1):
                continue
                
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss

# データ関連のユーティリティ関数
def remove_file_extention(file_name):
    # remove file extension (.txt) given filename
    return file_name.split('.')[0]

def add_file_extention(file_name, extention):
    # add file extension (.txt) given filename
    return file_name + '.' + extention

def clear_folder(path):
    # remove all files in the folder
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Folder successfully removed: ", path)
    else:
        print("No such path: ", path)

def delete_file(path, file_name_list):
    # delete given file list
    for file in file_name_list:
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print("File successfully deleted: ", file_path)
            else:
                print("Error: %s file not found" % file_path)        
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

def get_all_file_names(path):
    # return all file names given directory
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

def unique_list(l):
    # get unique elements from list
    x = []
    for a in l:
        if a not in x:
            x.append(a)
    return x

def angle_between(p1, p2):
    # return angle between two points
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ((ang1 - ang2) % (2 * np.pi))

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]
