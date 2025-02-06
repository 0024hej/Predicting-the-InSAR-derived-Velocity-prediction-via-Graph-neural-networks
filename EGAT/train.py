from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import EGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_heads', type=int, default=8,
                    help='Attention heads.')
parser.add_argument('--l_nums', type=int, default=3,
                    help='Number of EGATLayer.')
parser.add_argument('--lamda', type=float, default=0.25,
                    help='The node feature ratio.')
args =parser.parse_known_args()[0]

args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Load data
graph, node_features, edge_features, labels, idx_train, idx_val, idx_test = load_data()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:0')
# Define EGAT model and optimizer
model = EGAT(node_feats=node_features.shape[1],
            edge_feats=edge_features.shape[1],
            f_h = 128,
            f_e = 128,
            lamda = args.lamda,
            num_heads = args.num_heads,
            dropout=args.dropout,
            pred_hid = 128, 
            l_num = args.l_nums, 
             )
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    print("pppppppppppppppppppppppppppp")
    graph = graph.to(device)
    model = model.to(device)
    node_features = node_features.to(device)
    edge_features = edge_features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(graph, node_features, edge_features)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train]).to(device)
    print("loss------------------------------------------------------111")
    print(output.shape,labels.shape)
    print(output[idx_val], labels[idx_val])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(graph, node_features, edge_features)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val]).to(device)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(graph, node_features, edge_features)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test]).to(device)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
print("oooooooooooooooooooo")
print(graph.device)
#print(graph.is_cuda)
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
