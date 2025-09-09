import torch
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os
import time
import pickle
import subprocess

from model import SocialModel
from utils import DataLoader, get_mean_error, get_final_error, revert_seq, vectorize_seq, time_lr_scheduler, Gaussian2DLikelihood
from grid import getSequenceGridMask

def main():
    
    parser = argparse.ArgumentParser()
    
    # --- 変更点: データセットの場所を指定する引数を必須で追加 ---
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the datasets (e.g., /content/datasets)')

    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # Cuda parameter
    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
    # GRU parameter
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=2,
                        help='Total number of validation dataset for validate accuracy')
    # frequency of validation
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    parser.add_argument('--grid', action="store_true", default=True,
                        help='Whether store grids and use further epoch')
    parser.add_argument('--forcePreProcess', action="store_true", default=False,
                        help='Force preprocess the data again')

    args = parser.parse_args()
    
    train(args)


def train(args):
    # ログ・モデル保存用のディレクトリ作成
    os.makedirs("log", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # --- 変更点: DataLoaderにデータパス(f_prefix)を渡すように修正 ---
    dataloader = DataLoader(f_prefix=args.data_root, 
                            batch_size=args.batch_size, 
                            seq_length=args.seq_length, 
                            num_of_validation=args.num_validation, 
                            forcePreProcess=args.forcePreProcess)

    model_name = "LSTM"
    method_name = "SOCIALLSTM"
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    # 各種ディレクトリパスの設定
    log_directory = os.path.join('log', method_name, model_name)
    save_directory = os.path.join('model', method_name, model_name)
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(save_directory, exist_ok=True)

    # Logging files
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w+')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w+')
    
    # Save the arguments in the config file
    with open(os.path.join(save_directory,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, save_tar_name+str(x)+'.tar')

    # model creation
    net = SocialModel(args)
    if args.use_cuda:
        net = net.to(device)

    optimizer = torch.optim.Adagrad(net.parameters())
    
    # グリッドを保存するためのリスト
    grids = [[] for _ in range(dataloader.numDatasets)]

    # Training
    for epoch in range(args.num_epochs):
        print(f'****************Training epoch {epoch+1}/{args.num_epochs} beginning******************')
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()
            x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()
            
            # バッチが空の場合はスキップ
            if not x:
                continue

            loss_batch = 0

            # For each sequence
            for sequence in range(len(x)): # batch_sizeではなく取得できたシーケンス数でループ
                x_seq, _, d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                
                x_seq_tensor, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                
                # シーケンス内に歩行者がいない場合はスキップ
                if not lookup_seq:
                    continue

                folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                dataset_data = dataloader.get_dataset_dimension(folder_name)
                
                if args.grid:
                    if epoch == 0:
                        grid_seq = getSequenceGridMask(x_seq_tensor, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                        grids[d_seq].append(grid_seq)
                    else:
                        batch_index = (batch * args.batch_size) + sequence
                        if batch_index < len(grids[d_seq]):
                            grid_seq = grids[d_seq][batch_index]
                        else: # グリッドが保存されていない場合は再計算（安全策）
                            grid_seq = getSequenceGridMask(x_seq_tensor, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                else:
                    grid_seq = getSequenceGridMask(x_seq_tensor, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)

                vec_x_seq, _ = vectorize_seq(x_seq_tensor.clone(), PedsList_seq, lookup_seq)
                
                if args.use_cuda:
                    vec_x_seq = vec_x_seq.to(device)

                numNodes = len(lookup_seq)
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size)).to(device)
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size)).to(device)

                net.zero_grad()
                optimizer.zero_grad()
                
                outputs, _, _ = net(vec_x_seq, grid_seq, hidden_states, cell_states, PedsList_seq, numPedsList_seq, dataloader, lookup_seq)
                
                loss = Gaussian2DLikelihood(outputs, vec_x_seq, PedsList_seq, lookup_seq)
                loss_batch += loss.item()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()

            end = time.time()
            if len(x) > 0:
                loss_batch /= len(x)
                loss_epoch += loss_batch
            
            print(f'{epoch * dataloader.num_batches + batch + 1}/{args.num_epochs * dataloader.num_batches} (epoch {epoch+1}), '
                  f'train_loss = {loss_batch:.3f}, time/batch = {end - start:.3f}s')

        if dataloader.num_batches > 0:
            loss_epoch /= dataloader.num_batches
            log_file_curve.write(f"Training epoch: {epoch+1} loss: {loss_epoch}\n")

        # optimizerの学習率を調整
        optimizer = time_lr_scheduler(optimizer, epoch, lr_decay_epoch=args.freq_optimizer)

        # Save the model after each epoch
        if (epoch+1) % 5 == 0: # 5エポックごとに保存
            print('Saving model checkpoint')
            torch.save({
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch+1))

    print('Training finished.')
    log_file.close()
    log_file_curve.close()

if __name__ == '__main__':
    main()
