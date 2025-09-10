import numpy as np
import torch
from torch.autograd import Variable
import os
import math

class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)

def getCoef(outputs):
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr

def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]
    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    converted_node_present = [look_up.get(node) for node in nodesPresent if look_up.get(node) is not None]

    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        # Ensure gradients are not tracked for numpy conversion
        mean = [o_mux[node].item(), o_muy[node].item()]
        cov = [[o_sx[node].item()**2, o_corr[node].item()*o_sx[node].item()*o_sy[node].item()],
               [o_corr[node].item()*o_sx[node].item()*o_sy[node].item(), o_sy[node].item()**2]]
        
        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]
    return next_x.to(mux.device), next_y.to(mux.device)

def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, using_cuda, look_up):
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()
    for tstep in range(pred_length):
        counter = 0
        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)
            if nodeID not in trueNodesPresent[tstep] or lookup.get(nodeID) is None:
                continue
            
            node_idx = look_up[nodeID]
            # Ensure index is within bounds
            if node_idx >= ret_nodes.shape[1] or node_idx >= nodes.shape[1]:
                continue
                
            pred_pos = ret_nodes[tstep, node_idx, :]
            true_pos = nodes[tstep, node_idx, :]
            error[tstep] += torch.norm(pred_pos - true_pos, p=2)
            counter += 1
        if counter != 0:
            error[tstep] = error[tstep] / counter
    return torch.mean(error)

def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent[tstep]:
        nodeID = int(nodeID)
        if nodeID not in trueNodesPresent[tstep] or lookup.get(nodeID) is None:
            continue
        
        node_idx = look_up[nodeID]
        if node_idx >= ret_nodes.shape[1] or node_idx >= nodes.shape[1]:
            continue

        pred_pos = ret_nodes[tstep, node_idx, :]
        true_pos = nodes[tstep, node_idx, :]
        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1
    if counter != 0:
        error = error / counter
    return error

def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
    seq_length = outputs.size()[0]
    mux, muy, sx, sy, corr = getCoef(outputs)
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2
    result = torch.exp(-z/(2*negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    result = result / denom
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    loss = 0
    counter = 0
    for framenum in range(seq_length):
        nodeIDs = [int(nodeID) for nodeID in nodesPresent[framenum] if nodeID != -1]
        for nodeID in nodeIDs:
            node_idx = look_up.get(nodeID)
            if node_idx is not None and node_idx < result.shape[1]:
                loss = loss + result[framenum, node_idx]
                counter = counter + 1
    if counter != 0:
        return loss / counter
    else:
        return loss

def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            if ped == -1: continue
            ped_idx = lookup_seq.get(ped)
            if ped_idx is not None:
                first_values_dict[ped] = frame[ped_idx, 0:2]
                vectorized_x_seq[ind, ped_idx, 0:2] = frame[ped_idx, 0:2] - first_values_dict[ped][0:2]
    return vectorized_x_seq, first_values_dict
