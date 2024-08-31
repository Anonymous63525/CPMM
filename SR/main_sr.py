import os
import pdb
from SR.datasets_sr import SASRecDataset
from SR.utils_sr import predicting, EarlyStopping, get_user_seqs, set_seed
import numpy as np
import random
import torch
from torch.optim import Adam
from SR.sasrec import SASRec
import pickle
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--output_dir', default='output_SR/', type=str)
    parser.add_argument('--data_name', default='Toys', type=str)

    # parser.add_argument('--image_path', default='../outputs/align_features/align_image_features_Tools_pca_128.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/align_features/align_text_features_Tools_pca_128.pt', type=str)
    # parser.add_argument('--item_path', default=None)
    #
    # parser.add_argument('--image_path', default='../outputs/bt_image_features_Toys_pca_128_Toys_pca_128_features/bt_image_features_Toys_pca_128.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/bt_features/bt_text_features_Toys_pca_128.pt', type=str)
    # parser.add_argument('--item_path', default=None)

    # parser.add_argument('--image_path', default='../outputs/clip_features/clip_image_features_Toys_pca_128.pt', type=str)
    # parser.add_argument('--image_path', default='../outputs/clip_features/clip_image_features_Beauty_hust.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/clip_features/clip_text_features_Toys_pca_128.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/clip_features/clip_text_features_Beauty_hust.pt', type=str)
    # parser.add_argument('--item_path', default=None)
    #
    # parser.add_argument('--image_path', default='../outputs/blip_features/blip_image_features_Toys_pca_128.pt', type=str)
    # parser.add_argument('--image_path', default='../outputs/Beauty/blip_image_features_Beauty_pca_128.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/blip_features/blip_text_features_Toys_pca_128.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/Beauty/blip_text_features_Beauty_pca_128.pt', type=str)
    # parser.add_argument('--item_path', default=None)
    #
    # parser.add_argument('--image_path', default='../outputs/flava_features/flava_image_features_Tools_pca_128.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/flava_features/flava_text_features_Tools_pca_128.pt', type=str)
    # parser.add_argument('--item_path', default=None)

    # parser.add_argument('--image_path', default='../outputs/Beauty/project_image_features_Beauty_all_SR.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/Beauty/project_text_features_Beauty_all_SR.pt', type=str)
    # parser.add_argument('--item_path', default='../outputs/Beauty/project_ids_features_Beauty_all_SR.pt', type=str)
    # parser.add_argument('--item_path', default=None)

    # parser.add_argument('--image_path', default='../outputs/Beauty/project_image_features_Beauty_SR_.pt', type=str)
    # parser.add_argument('--text_path', default='../outputs/Beauty/project_text_features_Beauty_SR_.pt', type=str)
    # parser.add_argument('--item_path', default='../outputs/Beauty/project_ids_features_Beauty_SR_.pt', type=str)

    parser.add_argument('--image_path', default='outputs/Toys/project_image_features_Toys_SR_.pt', type=str)
    parser.add_argument('--text_path', default='outputs/Toys/project_text_features_Toys_SR_.pt', type=str)
    parser.add_argument('--item_path', default='outputs/Toys/project_ids_features_Toys_SR_.pt', type=str)

    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument("--pretraining", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding', default=True)
    parser.add_argument("--ban_bp", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--Add", type=lambda x: (str(x).lower() == 'true'), help='Whether to use pretraining embedding',default=True)
    parser.add_argument("--train_tv", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--proj", type=lambda x: (str(x).lower() == 'true'), default=False)

    # model args
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu_id")
    parser.add_argument("--id", type=int, default=5)

    # if

    args = parser.parse_args()

    set_seed(args.seed)

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    args.data_file = args.data_dir + args.data_name + '.txt'
    # item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    # matrix是训练集的协同过滤矩阵，用于在推荐的时候产生未推荐过的item
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 1
    args.num_users = num_users
    args.mask_id = max_item + 1
    # args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.kernel_param}-{args.pretraining}-{args.id}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    args.train_matrix = valid_rating_matrix
    args.test_matrix = test_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # user_id input_ids target_pos target_neg answer
    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.model_name == "SASRec":
        model = SASRec(args).to(args.device)

    betas = (args.adam_beta1, args.adam_beta2)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=False)
    start_time = time.time()
    for epoch in range(5000):
        model.train()
        rec_avg_loss = 0.0
        rec_cur_loss = 0.0

        for data in train_dataloader:
            optimizer.zero_grad()
            if args.model_name == 'BERT4Rec':
                interation = {
                    'MASK_ITEM_SEQ': data[0].to(args.device),  # user_id for testing
                    'POS_ITEMS': data[1].to(args.device),
                    'NEG_ITEMS': data[2].to(args.device),
                    'MASK_INDEX': data[3].to(args.device),
                }
            else:
                interation = {
                    'USER_ID': data[0].to(args.device),  # user_id for testing
                    'ITEM_SEQ': data[1].to(args.device),
                    'ITEM_SEQ_LEN': data[2].to(args.device),
                    'POS_ITEM_ID': data[3].to(args.device),
                    'NEG_ITEM_ID': data[4].to(args.device),
                }

            loss = model.calculate_loss(interation)
            loss.backward()
            optimizer.step()

            rec_avg_loss += loss.item()
            rec_cur_loss = loss.item()
        post_fix = {
            "epoch": epoch,
            "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(train_dataloader)),
            "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
        }
        print(post_fix)
        scores, _, _ = predicting(args, model, args.device, eval_dataloader, epoch, stage='val')
        data = {
            "HIT@1": '{:.8f}'.format(scores[0]), "NDCG@1": '{:.8f}'.format(scores[1]),
            "HIT@5": '{:.8f}'.format(scores[2]), "NDCG@5": '{:.8f}'.format(scores[3]),
            "HIT@10": '{:.8f}'.format(scores[4]), "NDCG@10": '{:.8f}'.format(scores[5]),
            "HIT@15": '{:.8f}'.format(scores[6]), "NDCG@15": '{:.8f}'.format(scores[7]),
            "HIT@20": '{:.8f}'.format(scores[8]), "NDCG@20": '{:.8f}'.format(scores[9]),
            # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(cndcg[5]),
            "MRR": '{:.8f}'.format(scores[-1])}
        print(data)
        early_stopping(np.array([scores[-2:-1]]), model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Training completed in {total_time} seconds.')
    test_model = SASRec(args).to(args.device)
    test_model.load_state_dict(torch.load(args.checkpoint_path))

    start_time = time.time()
    val_data, _, _ = predicting(args, test_model, args.device, eval_dataloader, epoch, stage='val')
    test_data, _, _ = predicting(args, test_model, args.device, test_dataloader, epoch, stage='test')
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Inference completed in {total_time} seconds.')
    val_post_fix = {
        "HIT@1": '{:.8f}'.format(val_data[0]), "NDCG@1": '{:.8f}'.format(val_data[1]),
        "HIT@5": '{:.8f}'.format(val_data[2]), "NDCG@5": '{:.8f}'.format(val_data[3]),
        "HIT@10": '{:.8f}'.format(val_data[4]), "NDCG@10": '{:.8f}'.format(val_data[5]),
        "HIT@15": '{:.8f}'.format(val_data[6]), "NDCG@15": '{:.8f}'.format(val_data[7]),
        "HIT@20": '{:.8f}'.format(val_data[8]), "NDCG@20": '{:.8f}'.format(val_data[9]),
        # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(cndcg[5]),
        "MRR": '{:.8f}'.format(val_data[-1])}
    test_post_fix = {
        "HIT@1": '{:.8f}'.format(test_data[0]), "NDCG@1": '{:.8f}'.format(test_data[1]),
        "HIT@5": '{:.8f}'.format(test_data[2]), "NDCG@5": '{:.8f}'.format(test_data[3]),
        "HIT@10": '{:.8f}'.format(test_data[4]), "NDCG@10": '{:.8f}'.format(test_data[5]),
        "HIT@15": '{:.8f}'.format(test_data[6]), "NDCG@15": '{:.8f}'.format(test_data[7]),
        "HIT@20": '{:.8f}'.format(test_data[8]), "NDCG@20": '{:.8f}'.format(test_data[9]),
        # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(cndcg[5]),
        "MRR": '{:.8f}'.format(test_data[-1])}
    print(val_post_fix, flush=True)
    print(test_post_fix, flush=True)
    return val_post_fix
# main()