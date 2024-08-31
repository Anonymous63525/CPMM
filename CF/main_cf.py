import pdb
from CF.lightgcn import LightGCN
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import time
import argparse
import os
import torch.optim as optim
import logging
from CF.utils_cf import EarlyStopping, predicting, get_user_seqs

# def main(device, learningrate):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', default='data/', type=str)
#     parser.add_argument('--output_dir', default='output_CF/', type=str)
#     parser.add_argument('--dataname', default='Tools', type=str)
#
#     parser.add_argument("--model_name", default='LightGCN', type=str)
#     parser.add_argument("--pretraining", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding', default=False)
#     parser.add_argument("--ban_bp", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding', default=False)
#     parser.add_argument("--proj", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding', default=False)
#     parser.add_argument("--lr", type=float, default=0.001)
#     parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
#     parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
#     parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
#     parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
#     parser.add_argument("--device", type=str, default="cuda:0")
#     parser.add_argument("--index", type=int, default=102)
#     parser.add_argument("--rank", type=int, default=101)
#
#     args = parser.parse_args()
#
#     args.device = device
#
#     args.image_path = f'outputs/{args.dataname}/project_image_features_{args.dataname}_CF_{learningrate}_.pt'
#     args.text_path = f'outputs/{args.dataname}/project_text_features_{args.dataname}_CF_{learningrate}_.pt'
#     args.item_path = f'outputs/{args.dataname}/project_ids_features_{args.dataname}_CF_{learningrate}_.pt'
#
#     print(f'model:{args.model_name},pretraining:{args.pretraining},weight_decay:{args.weight_decay}')
#     args_str = f'{args.model_name}-{args.dataname}-{args.hidden_size}-{args.lr}-{args.weight_decay}-{args.pretraining}-{args.index}'
#     checkpoint = args_str + '.pt'
#     args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
#
#     def set_seed(seed):
#         random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#
#
#     class YourDataset(Dataset):
#         def __init__(self, dataset):
#             self.dataset = dataset
#
#         def __len__(self):
#             return len(self.dataset)
#
#         def __getitem__(self, idx):
#             u_id = self.dataset.loc[idx, 'user_id']
#             i_id = self.dataset.loc[idx, 'item_id']
#
#             i_neg = i_id
#             while i_neg == i_id:
#                 i_neg = random.randint(1, args.max_item - 1)
#
#             return u_id, i_id, i_neg
#
#     args.data_file = f"./CF/dataset/{args.dataname}_train.csv"
#     valid_rating_matrix, num_users, num_items = get_user_seqs(args.data_file)
#     args.train_matrix = valid_rating_matrix
#     args.max_item = num_items
#     args.num_users = num_users
#
#     set_seed(42)
#     device = args.device
#     train_dataset = pd.read_csv(f"./CF/dataset/{args.dataname}_train.csv")
#     val_dataset = pd.read_csv(f"./CF/dataset/{args.dataname}_val.csv")
#     test_dataset = pd.read_csv(f"./CF/dataset/{args.dataname}_test.csv")
#
#     train_dataset = YourDataset(train_dataset)
#     val_dataset = YourDataset(val_dataset)
#     test_dataset = YourDataset(test_dataset)
#
#     train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
#
#     user_id = []
#     item_id = []
#     for batch in train_loader:
#         user_id += batch[0]
#         item_id += batch[1]
#     # 用于构建训练集的协同过滤矩阵
#     df_feat = {
#         'user_id':np.array(user_id),
#         'item_id':np.array(item_id)
#     }
#
#     early_stop = EarlyStopping(args.checkpoint_path, patience=30, verbose=True)
#
#     args.inter_feat = df_feat
#     args.uid_field = 'user_id'
#     args.iid_field = 'item_id'
#
#     if args.model_name == 'LightGCN':
#         model = LightGCN(args).to(device)
#
#
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     best_mse = 0
#     for epoch in range(5000):
#         model.train()
#         for data in train_loader:
#             interaction = {
#                 'user_id':data[0].to(device),
#                 'item_id':data[1].to(device),
#                 'neg_item_id':data[2].to(device)
#             }
#             optimizer.zero_grad()
#             cur_loss = model.calculate_loss(interaction)
#             cur_loss.backward()
#             optimizer.step()
#         print(cur_loss)
#
#         scores, _, _ = predicting(args, model, device, val_loader, epoch, stage='val')
#         # scores, _, _ = predicting(args, model, device, test_loader, epoch, stage='val')
#         early_stop(np.array([scores[-4]]), model)
#
#         if early_stop.early_stop:
#             print("Early stopping")
#             break
#
#     # model.load_state_dict(torch.load(args.checkpoint_path))
#
#     test_model = LightGCN(args).to(device)
#     test_model.load_state_dict(torch.load(args.checkpoint_path))
#
#     eval_data, _, _ = predicting(args, test_model, args.device, val_loader, epoch, stage='val')
#     test_data, _, _ = predicting(args, test_model, args.device, test_loader, epoch, stage='test')
#
#     # print(f'Inference completed in {total_time} seconds.')
#     post_fix_val = {'NOW valid:'
#                 "HIT@1": '{:.8f}'.format(eval_data[0]), "NDCG@1": '{:.8f}'.format(eval_data[1]),
#                 "HIT@5": '{:.8f}'.format(eval_data[2]), "NDCG@5": '{:.8f}'.format(eval_data[3]),
#                 "HIT@10": '{:.8f}'.format(eval_data[4]), "NDCG@10": '{:.8f}'.format(eval_data[5]),
#                 "HIT@15": '{:.8f}'.format(eval_data[6]), "NDCG@15": '{:.8f}'.format(eval_data[7]),
#                 "HIT@20": '{:.8f}'.format(eval_data[8]), "NDCG@20": '{:.8f}'.format(eval_data[9]),
#                 # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
#                 "MRR": '{:.8f}'.format(eval_data[-1])
#                 }
#     post_fix_test = {'NOW test:'
#                 "HIT@1": '{:.8f}'.format(test_data[0]), "NDCG@1": '{:.8f}'.format(test_data[1]),
#                 "HIT@5": '{:.8f}'.format(test_data[2]), "NDCG@5": '{:.8f}'.format(test_data[3]),
#                 "HIT@10": '{:.8f}'.format(test_data[4]), "NDCG@10": '{:.8f}'.format(test_data[5]),
#                 "HIT@15": '{:.8f}'.format(test_data[6]), "NDCG@15": '{:.8f}'.format(test_data[7]),
#                 "HIT@20": '{:.8f}'.format(test_data[8]), "NDCG@20": '{:.8f}'.format(test_data[9]),
#                 # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
#                 "MRR": '{:.8f}'.format(test_data[-1])
#                 }
#     logging.info(post_fix_val)
#     logging.info(post_fix_test)
#
#     return eval_data, test_data
# main()
# predicting(args, model, device, val_loader, epoch, stage='val')
# predicting(args, model, device, test_loader, epoch, stage='test')
import pdb
from CF.lightgcn import LightGCN
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import time
import argparse
import os
import torch.optim as optim
import logging
from CF.utils_cf import EarlyStopping, predicting, get_user_seqs

def main(device, learningrate):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--output_dir', default='output_CF/', type=str)
    parser.add_argument('--dataname', default='Toys', type=str)

    parser.add_argument("--model_name", default='LightGCN', type=str)
    parser.add_argument("--pretraining", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding', default=True)
    parser.add_argument("--ban_bp", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding', default=True)
    parser.add_argument("--proj", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding', default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--index", type=int, default=109)
    parser.add_argument("--rank", type=int, default=103)

    args = parser.parse_args()

    args.device = device

    args.image_path = f'outputs/{args.dataname}/project_image_features_{args.dataname}_CF_{learningrate}_.pt'
    args.text_path = f'outputs/{args.dataname}/project_text_features_{args.dataname}_CF_{learningrate}_.pt'
    args.item_path = f'outputs/{args.dataname}/project_ids_features_{args.dataname}_CF_{learningrate}_.pt'


    # args.image_path = f'../outputs/Beauty/clip_image_features_Beauty_pca_128.pt'
    # args.text_path = f'../outputs/Beauty/clip_text_features_Beauty_pca_128.pt'
    # args.item_path = None

    print(f'model:{args.model_name},pretraining:{args.pretraining},weight_decay:{args.weight_decay}')
    args_str = f'{args.model_name}-{args.dataname}-{args.hidden_size}-{args.lr}-{args.weight_decay}-{args.pretraining}-{args.index}'
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

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

            i_neg = i_id
            while i_neg == i_id:
                i_neg = random.randint(1, args.max_item - 1)

            return u_id, i_id, i_neg

    args.data_file = f"./CF/dataset/{args.dataname}_train.csv"
    valid_rating_matrix, num_users, num_items = get_user_seqs(args.data_file)
    args.train_matrix = valid_rating_matrix
    args.max_item = num_items
    args.num_users = num_users

    set_seed(42)
    device = args.device
    train_dataset = pd.read_csv(f"./CF/dataset/{args.dataname}_train.csv")
    val_dataset = pd.read_csv(f"./CF/dataset/{args.dataname}_val.csv")
    test_dataset = pd.read_csv(f"./CF/dataset/{args.dataname}_test.csv")

    train_dataset = YourDataset(train_dataset)
    val_dataset = YourDataset(val_dataset)
    test_dataset = YourDataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    user_id = []
    item_id = []
    for batch in train_loader:
        user_id += batch[0]
        item_id += batch[1]
    # 用于构建训练集的协同过滤矩阵
    df_feat = {
        'user_id':np.array(user_id),
        'item_id':np.array(item_id)
    }

    early_stop = EarlyStopping(args.checkpoint_path, patience=30, verbose=True)

    args.inter_feat = df_feat
    args.uid_field = 'user_id'
    args.iid_field = 'item_id'

    if args.model_name == 'LightGCN':
        model = LightGCN(args).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mse = 0
    for epoch in range(5000):
        model.train()
        for data in train_loader:
            interaction = {
                'user_id':data[0].to(device),
                'item_id':data[1].to(device),
                'neg_item_id':data[2].to(device)
            }
            optimizer.zero_grad()
            cur_loss = model.calculate_loss(interaction)
            cur_loss.backward()
            optimizer.step()
        print(cur_loss)

        scores, _, _ = predicting(args, model, device, val_loader, epoch, stage='val')
        # scores, _, _ = predicting(args, model, device, test_loader, epoch, stage='val')
        early_stop(np.array([scores[-4]]), model)

        if early_stop.early_stop:
            print("Early stopping")
            break

    # model.load_state_dict(torch.load(args.checkpoint_path))

    test_model = LightGCN(args).to(device)
    test_model.load_state_dict(torch.load(args.checkpoint_path))

    eval_data, _, _ = predicting(args, test_model, args.device, val_loader, epoch, stage='val')
    test_data, _, _ = predicting(args, test_model, args.device, test_loader, epoch, stage='test')

    # print(f'Inference completed in {total_time} seconds.')
    post_fix_val = {'NOW valid:'
                "HIT@1": '{:.8f}'.format(eval_data[0]), "NDCG@1": '{:.8f}'.format(eval_data[1]),
                "HIT@5": '{:.8f}'.format(eval_data[2]), "NDCG@5": '{:.8f}'.format(eval_data[3]),
                "HIT@10": '{:.8f}'.format(eval_data[4]), "NDCG@10": '{:.8f}'.format(eval_data[5]),
                "HIT@15": '{:.8f}'.format(eval_data[6]), "NDCG@15": '{:.8f}'.format(eval_data[7]),
                "HIT@20": '{:.8f}'.format(eval_data[8]), "NDCG@20": '{:.8f}'.format(eval_data[9]),
                # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
                "MRR": '{:.8f}'.format(eval_data[-1])
                }
    post_fix_test = {'NOW test:'
                "HIT@1": '{:.8f}'.format(test_data[0]), "NDCG@1": '{:.8f}'.format(test_data[1]),
                "HIT@5": '{:.8f}'.format(test_data[2]), "NDCG@5": '{:.8f}'.format(test_data[3]),
                "HIT@10": '{:.8f}'.format(test_data[4]), "NDCG@10": '{:.8f}'.format(test_data[5]),
                "HIT@15": '{:.8f}'.format(test_data[6]), "NDCG@15": '{:.8f}'.format(test_data[7]),
                "HIT@20": '{:.8f}'.format(test_data[8]), "NDCG@20": '{:.8f}'.format(test_data[9]),
                # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
                "MRR": '{:.8f}'.format(test_data[-1])
                }
    logging.info(post_fix_val)
    logging.info(post_fix_test)

    return eval_data, test_data
# main()
# predicting(args, model, device, val_loader, epoch, stage='val')
# predicting(args, model, device, test_loader, epoch, stage='test')
