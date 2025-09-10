import argparse
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pickle

from model import SocialModel
from utils import TrajectoryDataset
from grid import getSequenceGridMask
import torch.distributions.multivariate_normal as torchdist

def ade_fde(pred_traj, true_traj):
    """
    Average Displacement Error (ADE)とFinal Displacement Error (FDE)を計算する。
    
    pred_traj: 予測された軌道 (pred_len, num_peds, 2)
    true_traj: 正解の軌道 (pred_len, num_peds, 2)
    """
    error = torch.sqrt(((pred_traj - true_traj)**2).sum(dim=2))
    ade = error.mean()
    fde = error[-1, :].mean()
    return ade, fde

def test(model, loader, args, device, k_steps=20):
    """
    学習済みモデルをテストデータで評価する関数。
    Best-of-Nサンプリングを用いてADEとFDEを計算する。
    """
    model.eval()
    ade_list, fde_list = [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, V_obs, A_obs, V_tr, A_tr) = batch

            obs_traj = obs_traj.squeeze(0).to(device)
            pred_traj_gt = pred_traj_gt.squeeze(0).to(device)
            obs_traj_rel = obs_traj_rel.squeeze(0).to(device)
            
            num_peds = obs_traj.size(1)
            
            batch_ade_list, batch_fde_list = [], []

            for _ in range(k_steps):
                pred_traj_fake_rel = torch.zeros(args.pred_len, num_peds, 2).to(device)
                
                grids_obs = []
                for t in range(args.obs_len):
                    grid = getSequenceGridMask(obs_traj[t], args.neighborhood_size, args.grid_size, args.use_cuda)
                    grids_obs.append(grid)
                
                hidden_states = torch.zeros(num_peds, args.rnn_size).to(device)
                cell_states = torch.zeros(num_peds, args.rnn_size).to(device)
                peds_list_obs = [torch.arange(num_peds).to(device) for _ in range(args.obs_len)]
                lookup = {i: i for i in range(num_peds)}
                
                _, hidden_states, cell_states = model(obs_traj_rel, grids_obs, hidden_states, cell_states, peds_list_obs, lookup)

                last_pos_abs = obs_traj[-1]
                last_pos_rel = obs_traj_rel[-1]

                for t in range(args.pred_len):
                    grid_pred = [getSequenceGridMask(last_pos_abs, args.neighborhood_size, args.grid_size, args.use_cuda)]
                    peds_list_pred = [torch.arange(num_peds).to(device)]
                    
                    output, hidden_states, cell_states = model(last_pos_rel.unsqueeze(0), grid_pred, hidden_states, cell_states, peds_list_pred, lookup)

                    muX = output[0, :, 0]
                    muY = output[0, :, 1]
                    sigX = torch.exp(output[0, :, 2])
                    sigY = torch.exp(output[0, :, 3])
                    rho = torch.tanh(output[0, :, 4])
                    
                    cov = torch.zeros(num_peds, 2, 2).to(device)
                    cov[:, 0, 0] = sigX * sigX
                    cov[:, 0, 1] = rho * sigX * sigY
                    cov[:, 1, 0] = rho * sigX * sigY
                    cov[:, 1, 1] = sigY * sigY
                    mean = output[0, :, 0:2]
                    
                    mvnormal = torchdist.MultivariateNormal(mean, cov)
                    next_pos_rel = mvnormal.sample()
                    
                    last_pos_abs += next_pos_rel
                    last_pos_rel = next_pos_rel
                    
                    pred_traj_fake_rel[t] = next_pos_rel
                
                pred_traj_fake_abs = torch.cumsum(pred_traj_fake_rel, dim=0) + obs_traj[-1].unsqueeze(0)
                
                ade, fde = ade_fde(pred_traj_fake_abs, pred_traj_gt)
                batch_ade_list.append(ade)
                batch_fde_list.append(fde)

            ade_list.append(min(batch_ade_list))
            fde_list.append(min(batch_fde_list))

    final_ade = torch.mean(torch.tensor(ade_list))
    final_fde = torch.mean(torch.tensor(fde_list))
    
    return final_ade, final_fde

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint directory (e.g., ./checkpoint/social_lstm_model/)')
    parser.add_argument('--k_steps', type=int, default=20, help='Number of samples for Best-of-N evaluation')
    
    main_args = parser.parse_args()

    if not os.path.exists(main_args.checkpoint_path):
        print(f"Error: Checkpoint path '{main_args.checkpoint_path}' does not exist.")
        return

    args_path = os.path.join(main_args.checkpoint_path, 'args.pkl')
    model_path = os.path.join(main_args.checkpoint_path, 'val_best.pth')

    if not os.path.exists(args_path) or not os.path.exists(model_path):
        print(f"Error: 'args.pkl' or 'val_best.pth' not found in the checkpoint directory.")
        return

    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    print(f"Using device: {device}")

    test_datasets = []
    print("Loading test dataset...")
    datasets_to_load = ['eth', 'hotel', 'univ', 'zara1', 'zara2'] if args.dataset == 'all' else [args.dataset]
    for name in datasets_to_load:
        test_path = f'./datasets/{name}/test/'
        if os.path.exists(test_path):
            dset = TrajectoryDataset(test_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1, delim='\t')
            test_datasets.append(dset)
    
    if not test_datasets:
        print("No test data found. Exiting.")
        return

    dset_test = ConcatDataset(test_datasets)
    loader_test = DataLoader(dset_test, batch_size=1, shuffle=False)
    print(f"Total test sequences: {len(dset_test)}")

    model = SocialModel(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("*"*30)
    print("Testing initiated...")
    ade, fde = test(model, loader_test, args, device, k_steps=main_args.k_steps)
    print("*"*30)
    print(f"Results for model: {main_args.checkpoint_path}")
    print(f"ADE (best of {main_args.k_steps}): {ade:.4f}")
    print(f"FDE (best of {main_args.k_steps}): {fde:.4f}")
    print("*"*30)

if __name__ == '__main__':
    main()
