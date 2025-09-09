import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import os
import time
import pickle

from model import SocialModel
from utils import TrajectoryDataset, vectorize_seq, Gaussian2DLikelihood, time_lr_scheduler
from grid import getSequenceGridMask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    
    # --- データセットとシーケンスに関する引数 ---
    parser.add_argument('--dataset_name', type=str, default='eth', help='Dataset name (eth, hotel, zara1, etc.)')
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observation length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Prediction length')
    
    # --- モデルのハイパーパラメータ ---
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--neighborhood_size', type=int, default=32, help='Neighborhood size')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size')
    parser.add_argument('--gru', action="store_true", default=False, help='Use GRU instead of LSTM')

    # --- 学習に関する引数 ---
    parser.add_argument('--batch_size', type=int, default=1, help='minibatch size (must be 1 for this model)')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    parser.add_argument('--lambda_param', type=float, default=0.0005, help='L2 regularization parameter')
    parser.add_argument('--freq_optimizer', type=int, default=8, help='Frequency of learning rate decay')

    # --- その他 ---
    parser.add_argument('--use_cuda', action="store_true", default=True, help='Use GPU or not')
    parser.add_argument('--tag', default='social_lstm_tag', help='personal tag for the model ')
    
    # SocialModelが必要とする固定の引数を設定
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--maxNumPeds', type=int, default=100) 
    
    args = parser.parse_args()
    
    args.seq_length = args.obs_length + args.pred_length
    train(args)

def train(args):
    # データセットのパスを組み立て
    # このスクリプト(train.py)と同じ階層にあるdatasetsフォルダを想定
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_root = os.path.join(script_dir, 'datasets', args.dataset_name)
    train_path = os.path.join(data_root, 'train')
    
    print(f"Loading training data from {train_path}...")
    train_dataset = TrajectoryDataset(data_dir=train_path, obs_len=args.obs_length, pred_len=args.pred_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Found {len(train_dataset)} sequences in the training set.")

    # モデルの準備
    net = SocialModel(args)
    if args.use_cuda:
        net = net.to(device)
    
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    
    # 保存用ディレクトリの準備
    model_name = "GRU" if args.gru else "LSTM"
    checkpoint_dir = os.path.join('./checkpoint', args.tag, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Training
    for epoch in range(args.num_epochs):
        net.train()
        loss_epoch = 0
        
        for batch_idx, (abs_traj, mask) in enumerate(train_loader):
            # DataLoaderはバッチ次元を追加するので、それを削除 (batch_size=1なので)
            abs_traj = abs_traj.squeeze(0) # (num_peds, seq_len, 2)
            mask = mask.squeeze(0) # (num_peds, seq_len)
            
            if args.use_cuda:
                abs_traj = abs_traj.to(device)
                mask = mask.to(device)
                
            # 絶対座標を相対座標に変換
            rel_traj, _ = vectorize_seq(abs_traj)
            
            # --- Social-LSTMモデルへの入力準備 ---
            num_peds = rel_traj.shape[0]
            
            # モデルの入力形式に合わせる (seq_len, num_peds, 2)
            full_rel_traj = rel_traj.permute(1, 0, 2)
            
            # モデルは古い形式のPedsListとlookup_seqを必要とするため、ダミーを作成
            peds_list_seq = [list(range(num_peds))] * args.seq_length
            lookup_seq = {i: i for i in range(num_peds)}

            # グリッドマスクの計算
            abs_traj_for_grid = abs_traj.permute(1, 0, 2)
            grid_seq = getSequenceGridMask(abs_traj_for_grid, [720, 576], peds_list_seq, args.neighborhood_size, args.grid_size, args.use_cuda)

            # 隠れ状態の初期化
            hidden_states = Variable(torch.zeros(num_peds, args.rnn_size)).to(device)
            cell_states = Variable(torch.zeros(num_peds, args.rnn_size)).to(device)

            # --- フォワードパスとロス計算 ---
            optimizer.zero_grad()
            
            # Social-LSTMはシーケンス全体を入力とする
            # 入力はt=0からt=18、ターゲットはt=1からt=19
            outputs, _, _ = net(full_rel_traj[:-1], grid_seq[:-1], hidden_states, cell_states, peds_list_seq[:-1], None, None, lookup_seq)
            
            # 損失計算 (ターゲットは1フレームずれる)
            loss_mask = mask.permute(1,0)[1:] # (seq_len-1, num_peds)
            loss = Gaussian2DLikelihood(outputs, full_rel_traj[1:], loss_mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            
            loss_epoch += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = loss_epoch / len(train_loader)
        print(f'---- Epoch [{epoch+1}/{args.num_epochs}] summary: Average Loss: {avg_loss:.4f} ----')
        
        # 学習率の減衰
        optimizer = time_lr_scheduler(optimizer, epoch, lr_decay_epoch=args.freq_optimizer)
        
        # モデルの保存
        if (epoch+1) % 10 == 0:
            print('Saving model checkpoint...')
            torch.save({
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f'model_{epoch+1}.tar'))

    print('Training finished.')

if __name__ == '__main__':
    main()
