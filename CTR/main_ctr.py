import pdb
import numpy as np
from xDeepFM import xDeepFM
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import argparse
import os
from utils_ctr import predicting
import torch.optim as optim
from utils_ctr import EarlyStopping, get_user_seqs


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='DCN', type=str)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--pretraining", type=str, help='Whether to use pretraining representations', default='True')
parser.add_argument("--weight_decay", type=float, default=0.000001)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batchsize", type=int, default=512)
parser.add_argument("--dataname", type=str, default='Beauty')

# pretraining representations path
parser.add_argument('--image_path', default='../outputs/Beauty/project_image_features_Beauty_CTR.pt', type=str)
parser.add_argument('--text_path', default='../outputs/Beauty/project_text_features_Beauty_CTR.pt', type=str)
parser.add_argument('--item_path', default='../outputs/Beauty/project_ids_features_Beauty_CTR.pt', type=str)

parser.add_argument("--embedding_size", type=int, default=128, help="hidden size of Embedding")
parser.add_argument("--num_feature_field", type=int, default=3, help="hidden size of FC")
parser.add_argument("--gpu_id", type=int, default=0)


args = parser.parse_args()

print(f'model:{args.model_name},pre_id:{args.pre_id},weight_decay:{args.weight_decay},dataname:{args.dataname},ablation:{args.ablation}')

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class YourDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        u_id = self.dataset.loc[idx, 'user_id']
        i_id = self.dataset.loc[idx, 'item_id']
        i_class = self.dataset.loc[idx, 'class']
        label = self.dataset.loc[idx, 'overall']
        return u_id, i_id, i_class, label

if args.dataname == 'Beauty':
    args.class_num = 7
elif args.dataname == 'Tools':
    args.class_num = 19
elif args.dataname == 'Toys':
    args.class_num = 25

args.data_file = args.dataname + '.txt'
user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(f'../data/{args.data_file}')
args.max_item = max_item + 1
args.num_users = num_users + 1

set_seed(42)
file_path = f'{args.dataname}_CTR.csv'
train_dataset = pd.read_csv(f"train_set_{args.dataname}_CTR.csv")
test_dataset = pd.read_csv(f"test_set_{args.dataname}_CTR.csv")
train_dataset = YourDataset(train_dataset)
test_dataset = YourDataset(test_dataset)

test_size = (int)(len(test_dataset) * 0.5)
val_dataset, test_dataset = torch.utils.data.random_split(
    dataset=test_dataset,
    lengths=[len(test_dataset) - test_size, test_size],
    generator=torch.Generator().manual_seed(0)
)

train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
checkpoint = f'output_ctr/{args.model_name}_{args.pre_id}_{args.ablation}_{args.lr}_{args.dropout_prob}_{args.weight_decay}_model.pth'

if args.model_name == 'xDeepFM':
    model = xDeepFM(args).to(device)

early_stop = EarlyStopping(checkpoint, patience=7, verbose=True)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
best_mse = 0
for epoch in range(1000):
    model.train()
    total_loss = 0
    batch_num = 0
    for data in train_loader:
        batch_num += 1
        optimizer.zero_grad()
        cur_loss = model.calculate_loss(args, data, device)
        cur_loss.backward()
        optimizer.step()
        total_loss += cur_loss
    print(total_loss / len(train_loader))

    average_loss, average_auc = predicting(args, model, device, val_loader, epoch, stage='val')
    early_stop(np.array([average_auc]), model)
    if early_stop.early_stop:
        print("Early stopping")
        break

model.load_state_dict(torch.load(checkpoint))
average_loss_test, average_auc_test = predicting(args, model, device, test_loader, epoch, stage='test')
