import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bivariate_loss(V_pred, V_gt):
    """
    二変量ガウス分布の負の対数尤度損失を計算する
    Args:
        V_pred (torch.Tensor): モデルの出力 (batch, seq, peds, 5)
        V_gt (torch.Tensor): 正解の相対座標 (batch, seq, peds, 2)
    """
    # 予測から5つのパラメータを分離
    mux, muy, sx, sy, corr = V_pred[:,:,:,0], V_pred[:,:,:,1], V_pred[:,:,:,2], V_pred[:,:,:,3], V_pred[:,:,:,4]

    # 活性化関数を適用
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)

    normx = V_gt[:,:,:,0] - mux
    normy = V_gt[:,:,:,1] - muy
    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # 負の対数尤度を計算
    result = torch.exp(-z/(2*negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    result = result / denom
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))

    # NaN（パディング）でない有効なデータのみで平均損失を計算
    mask = ~torch.isnan(V_gt[:,:,:,0])
    if not mask.any(): return torch.tensor(0.0)
    
    loss = torch.mean(result[mask])
    return loss

def sample_from_bivariate_gaussian(V_pred):
    """二変量ガウス分布から1点をサンプリングする"""
    mux, muy, sx, sy, corr = V_pred[:,:,:,0], V_pred[:,:,:,1], V_pred[:,:,:,2], V_pred[:,:,:,3], V_pred[:,:,:,4]
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)

    batch, seq_len, num_peds, _ = V_pred.shape
    
    # 乱数を生成
    rand_x = torch.randn(batch, seq_len, num_peds).to(device)
    rand_y = torch.randn(batch, seq_len, num_peds).to(device)
    
    # サンプリング式
    x = mux + sx * rand_x
    y = muy + sy * (corr * rand_x + torch.sqrt(1 - corr**2) * rand_y)
    
    return torch.stack([x, y], dim=-1)

def calculate_ade_fde(pred_abs, gt_abs):
    """ADEとFDEを計算する"""
    # NaNでない有効なデータのみを対象とするマスク
    mask = ~torch.isnan(gt_abs[:, :, :, 0])
    if not mask.any(): return 0.0, 0.0

    # L2ノルム（ユークリッド距離）を計算
    error = torch.norm(pred_abs - gt_abs, p=2, dim=-1)
    
    # ADE: 全フレームの平均誤差
    ade = torch.mean(error[mask])
    
    # FDE: 最終フレームの平均誤差
    final_mask = mask[:, -1, :]
    if not final_mask.any(): return ade.item(), 0.0
    fde = torch.mean(error[:, -1, :][final_mask])
    
    return ade.item(), fde.item()
