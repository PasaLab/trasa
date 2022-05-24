import torch
import argparse
import random
import pickle
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from train import TrainRunner
from data import YooDataset, read_dataset, CustomDataset
from collate import collate_fn
from model import TRASA

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# training configuration
parser.add_argument('--dataset-dir',    type=str,   default='data/diginetica', help='the dataset directory')
parser.add_argument('--device',         type=str,   default='cuda:0',          help='the device for training')
parser.add_argument('--batch-size',     type=int,   default=512,               help='the batch size for training')
parser.add_argument('--num-workers',    type=int,   default=4,                 help='the number of processes to load the data')
parser.add_argument('--epochs',         type=int,   default=20,                help='the number of training epochs')
parser.add_argument('--learning-rate',  type=float, default=0.01)
parser.add_argument('--lr-dc-step',     type=int,   default=3)
parser.add_argument('--patience',       type=int,   default=3)


# relation encoder
parser.add_argument('--rel-dim',         type=int, default=32)
parser.add_argument('--rnn-hidden-size', type=int, default=64)
parser.add_argument('--rnn-num-layers',  type=int, default=1)

# core architecture
parser.add_argument('--graph-layers', type=int, default=2)  
# parser.add_argument('--seq-layers',   type=int, default=1)
parser.add_argument('--embed-dim',    type=int, default=64)
parser.add_argument('--ff-embed-dim', type=int, default=128)
parser.add_argument('--num-heads',    type=int, default=4)

parser.add_argument('--dropout',      type=float, default=0.2)
parser.add_argument('--weight-decay', type=float, default=0.00001)
args = parser.parse_args()
print(args)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2022)

def main():
    dataset_dir = Path(args.dataset_dir)
    if args.dataset_dir == 'data/yoochoose1_4' or args.dataset_dir == 'data/yoochoose1_64':
        train_data = pickle.load(open(dataset_dir / 'train.txt', 'rb'))
        test_data = pickle.load(open(dataset_dir / 'test.txt', 'rb'))
        item_num = 37484
        train_set = YooDataset(train_data)
        test_set = YooDataset(test_data)
    else:
        train_sessions, test_sessions, item_num = read_dataset(dataset_dir)
        train_set = CustomDataset(train_sessions)
        test_set = CustomDataset(test_sessions)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    model = TRASA(
        args.embed_dim, item_num,
        args.rel_dim, args.rnn_hidden_size, args.rnn_num_layers,
        args.graph_layers, args.ff_embed_dim, args.num_heads,
        # args.seq_layers,
        args.dropout, args.device,
    )
    runner = TrainRunner(model, train_loader, test_loader, 
                         args.epochs, args.learning_rate, args.lr_dc_step, args.weight_decay, 
                         args.patience, args.device)
    print('start training')
    runner.train()
    

if __name__ == '__main__':
    main()