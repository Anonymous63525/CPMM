import pdb
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import random
import time
import os
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from loguru import logger
from utils import get_user_seqs, set_seed
from itertools import combinations
from tqdm import tqdm
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from SR.main_sr import main as sasrec
from CF.main_cf import main as lightgcn
from SR.datasets_sr import SASRecDataset
from torch.utils.data import DataLoader, SequentialSampler
import logging


class YourDataset(Dataset):
    def __init__(self, image_features_list, text_features_list, item_features_list):
        self.image_features_list = image_features_list
        self.text_features_list = text_features_list
        self.item_features_list = item_features_list

    def __len__(self):
        return len(self.image_features_list)

    def __getitem__(self, idx):
        img = self.image_features_list[idx]
        text = self.text_features_list[idx]
        ids = self.item_features_list[idx]
        # 加载图像
        return img, text, ids, idx + 1


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU(inplace=True)


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    recall_dict = {}
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            # sum_recall += len(act_set & pred_set) / float(len(act_set))
            one_user_recall = len(act_set & pred_set) / float(len(act_set))
            recall_dict[i] = one_user_recall
            sum_recall += one_user_recall
            true_users += 1
    return sum_recall / true_users, recall_dict


def cal_mrr(actual, predicted):
    sum_mrr = 0.
    true_users = 0
    num_users = len(predicted)
    mrr_dict = {}
    for i in range(num_users):
        r = []
        act_set = set(actual[i])
        pred_list = predicted[i]
        for item in pred_list:
            if item in act_set:
                r.append(1)
            else:
                r.append(0)
        r = np.array(r)
        if np.sum(r) > 0:
            # sum_mrr += np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            one_user_mrr = np.reciprocal(np.where(r == 1)[0] + 1, dtype=float)[0]
            sum_mrr += one_user_mrr
            true_users += 1
            mrr_dict[i] = one_user_mrr
        else:
            mrr_dict[i] = 0.
    return sum_mrr / len(predicted), mrr_dict


def ndcg_k(actual, predicted, topk):
    res = 0
    ndcg_dict = {}
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
        ndcg_dict[user_id] = dcg_k / idcg
    return res / float(len(actual)), ndcg_dict


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


# class MLP(nn.Module):
#     def __init__(self, channel=512, res_expansion=1, bias=True, activation='relu'):
#         super(MLP, self).__init__()
#         self.act = get_activation(activation)
#         self.net1 = nn.Sequential(
#             nn.Linear(channel, int(channel * res_expansion), bias=bias),
#             #nn.BatchNorm1d(int(channel * res_expansion)),
#             self.act
#         )
#         self.net2 = nn.Sequential(
#             nn.Linear(int(channel * res_expansion), int(channel * res_expansion), bias=bias),
#             nn.BatchNorm1d(int(channel * res_expansion))
#         )
#
#     def forward(self, x):
#         return self.net2(self.net1(x))

class MLP(nn.Module):
    def __init__(self, channel=512, res_expansion=1, bias=True, activation='relu'):
        super(MLP, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(channel, int(channel * res_expansion), bias=bias),
            # nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(channel * res_expansion), int(channel * res_expansion), bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion))
            # self.act
        )

    def forward(self, x):
        return self.net2(self.net1(x))


class CPMM(nn.Module):
    def __init__(self):
        super(CPMM, self).__init__()
        self.clip_project = MLP(res_expansion=0.25)
        self.clip_project3 = MLP(res_expansion=0.25)
        self.ids_project = MLP(res_expansion=0.25)
        self.ids_project3 = MLP(res_expansion=0.25)
        self.clip_project2 = MLP(channel=128)
        self.fuse_project2 = MLP(channel=128)

        # self.clip_project = MLP()
        # self.ids_project = MLP()
        # self.clip_project2 = MLP()
        # self.fuse_project2 = MLP()

    def finetune(self, item_features_list, text_features_list, image_features_list, std, device, mode='train'):
        if mode == 'train':
            image = F.normalize(image_features_list + (torch.randn(image_features_list.size()) * std).to(device))
            text = F.normalize(text_features_list + (torch.randn(text_features_list.size()) * std).to(device))
            ids = F.normalize(item_features_list + (torch.randn(item_features_list.size()) * std).to(device))

        elif mode == 'inference':
            image = F.normalize(image_features_list)
            text = F.normalize(text_features_list)
            ids = F.normalize(item_features_list)

        image = self.clip_project(image)
        image = self.clip_project2(image) + self.clip_project3(image_features_list)

        text = self.clip_project(text)
        text = self.clip_project2(text) + self.clip_project3(text_features_list)

        ids = self.ids_project(ids)
        ids = self.fuse_project2(ids) + self.ids_project3(item_features_list)

        image = F.normalize(image)
        text = F.normalize(text)
        ids = F.normalize(ids)
        return image, text, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='Toys', type=str)  # Beauty  Tools  Toys
    parser.add_argument('--recommendation_task', default='CF', type=str)  # SR CF CTR
    parser.add_argument('--std', default=0.002, type=float)
    parser.add_argument('--lamda1', default=2, type=float)  # 对比学习
    parser.add_argument('--lamda2', default=1, type=float)  # 模态间隙
    parser.add_argument('--lamda3', default=1, type=float)  # 协同信息
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epoch', default=200, type=float)
    parser.add_argument('--batchsize', default=128, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument("--rank", default=0, type=int)
    args = parser.parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        filename=f'{args.recommendation_task}/log/Ablation-{args.dataname}_{args.recommendation_task}_{args.rank}.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 设置时间格式
        )

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(f'data/{args.dataname}.txt')

    args.item_size = max_item + 1
    args.num_users = num_users

    args.train_matrix = valid_rating_matrix
    args.test_matrix = test_rating_matrix


    if args.recommendation_task == 'CTR':
        dense_matrix = np.zeros((args.num_users, args.item_size))
        df = pd.read_csv(f'train_set_{args.dataname}_CTR.csv')
        for index, row in tqdm(df.iterrows()):
            if int(row['overall']) == 1:
                dense_matrix[int(row['user_id']) - 1, int(row['item_id'])] = 1

    elif args.recommendation_task == 'CF':
        dense_matrix = np.zeros((args.num_users, args.item_size))
        df = pd.read_csv(f'./CF/dataset/{args.dataname}_train.csv')
        for index, row in tqdm(df.iterrows()):
            dense_matrix[int(row['user_id']) - 1, int(row['item_id'])] = 1

    elif args.recommendation_task == 'SR':
        dense_matrix = valid_rating_matrix.toarray()


    src = []
    dst = []
    for row in dense_matrix:
        item_id = np.nonzero(row)[0]
        combinations_list = list(combinations(item_id, 2))
        for node_pair in combinations_list:
            src.append(node_pair[0])
            dst.append(node_pair[1])


    edges1 = torch.cat((torch.tensor(src), torch.tensor(dst)), dim=0)
    edges2 = torch.cat((torch.tensor(dst), torch.tensor(src)), dim=0)
    num_nodes = max(max(src), max(dst)) + 1
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix[edges1, edges2] = 1
    adj_matrix += torch.eye(num_nodes)

    device = args.device if torch.cuda.is_available() else 'cpu'
    text_features_list = torch.load(f'outputs/clip_features/clip_text_features_{args.dataname}.pt').to(device)
    image_features_list = torch.load(f'outputs/clip_features/clip_image_features_{args.dataname}.pt').to(device)
    item_features_list = text_features_list + image_features_list

    dataset = YourDataset(image_features_list, text_features_list, item_features_list)

    logging.info(
        f"Learning Rate: {args.lr},Batch Size: {args.batchsize},Lambda 1: {args.lamda1},Lambda 2:{args.lamda2},Lambda 3: {args.lamda3}")
    set_seed(args.seed)
    model = CPMM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_ids = nn.BCEWithLogitsLoss()

    for epoch in range(args.epoch):
        batch_num = 0
        model.train()
        for data in dataset:
            contrastive_matrix = adj_matrix[data[-1], :][:, data[-1]].to(device)
            batch_num += 1
            optimizer.zero_grad()

            image_features_list1, text_features_list1, item_features_list1 = data[0], data[1], data[
                2]
            image, text, ids = model.finetune(item_features_list1, text_features_list1,
                                              image_features_list1, args.std, device, mode='train')
            ground_truth = torch.arange(len(data[0]), dtype=torch.long, device=device)

            logits_per_image_text = image @ text.t() * 100
            logits_per_text_image = logits_per_image_text.t()
            logits_per_ids = ids @ ids.t() * 100
            sim_text_fuse = torch.mean(torch.norm(text - ids, p='fro', dim=1))
            sim_image_fuse = torch.mean(torch.norm(image - ids, p='fro', dim=1))
            cur_loss = args.lamda1 * (loss_img(logits_per_image_text, ground_truth) + loss_txt(
                logits_per_text_image, ground_truth)) / 2 + \
                       args.lamda2 * (sim_text_fuse + sim_image_fuse) / 2 + args.lamda3 * loss_ids(
                logits_per_ids, contrastive_matrix)
            cur_loss.backward()
            optimizer.step()

    model.eval()
    # print('--------------Loading model parameters-------------------')
    # model.load_state_dict(torch.load(f"outputs/models/project_model_{args.dataname}_{args.recommendation_task}.pth"))
    image, text, ids = model.finetune(item_features_list, text_features_list, image_features_list,
                                      args.std, device, mode='inference')
    # print('-----------------Saving the results-------------------')
    os.makedirs(f'outputs/{args.dataname}', exist_ok=True)
    torch.save(text,
               f'outputs/{args.dataname}/project_text_features_{args.dataname}_{args.recommendation_task}_{args.lr}_.pt')
    torch.save(image,
               f'outputs/{args.dataname}/project_image_features_{args.dataname}_{args.recommendation_task}_{args.lr}_.pt')
    torch.save(ids,
               f'outputs/{args.dataname}/project_ids_features_{args.dataname}_{args.recommendation_task}_{args.lr}_.pt')


main()