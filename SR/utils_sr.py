import pdb
from logging import getLogger
import random
import os
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.sparse import csr_matrix

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

class AbstractRecommender(nn.Module):
    r"""Base class for all models"""

    def __init__(self):
        self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )

class SequentialRecommender(AbstractRecommender):
    def __init__(self, args):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        # self.USER_ID = config["USER_ID_FIELD"]
        # self.ITEM_ID = config["ITEM_ID_FIELD"]
        #
        self.ITEM_SEQ = 'ITEM_SEQ'
        self.ITEM_SEQ_LEN = 'ITEM_SEQ_LEN'
        self.POS_ITEM_ID = 'POS_ITEM_ID'
        self.NEG_ITEM_ID = 'NEG_ITEM_ID'

        self.max_seq_length = args.max_seq_length

        # load parameters info
        self.device = args.gpu_id

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # 创建一个基于压缩稀疏行（Compressed Sparse Row, CSR）格式的评分矩阵（rating matrix）的操作
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        # 将items中的元素添加到item_set中(合并两个集合)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 1

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        patience (int): 在验证损失（或其他性能指标）不再改善的情况下，等待多少个训练周期（或称为"epochs"）之后才触发早停。
                            Default: 7
        verbose (bool): 参数是一个布尔值，如果设置为True，那么在每次验证损失（或性能指标）有所改善时都会打印一条消息。
                            Default: False
        delta (float): delta 参数表示被认为是性能改善的最小阈值。如果验证损失（或性能指标）的改善小于 delta，则不会被视为足够显著的改善，不会重置耐心计数器。只有当验证损失改善大于 delta 时，才会重置耐心计数器。
                            Default: 0
        """
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model, epoch):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            # print('Save')
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def predicting(args, model, device, rec_data_iter, epoch, stage='val'):
    model.eval()

    answer_list = None
    # for i, batch in rec_data_iter:
    i = 0
    for batch in rec_data_iter:
        interation = {
            'USER_ID': batch[0].to(device),  # user_id for testing
            'ITEM_SEQ': batch[1].to(device),
            'ITEM_SEQ_LEN': batch[2].to(device),
            'POS_ITEM_ID': batch[3].to(device),
            'NEG_ITEM_ID': batch[4].to(device)
        }
        # 0. batch_data will be sent into the device(GPU or cpu)
        rating_pred = model.full_sort_predict(interation)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = interation['USER_ID'].cpu().numpy()
        if stage == 'val':
            rating_pred[args.train_matrix[batch_user_index].toarray() > 0] = 0
        else:
            rating_pred[args.test_matrix[batch_user_index].toarray() > 0] = 0
        ind = np.argpartition(rating_pred, -20)[:, -20:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if i == 0:
            pred_list = batch_pred_list
            answer_list = interation['POS_ITEM_ID'].unsqueeze(1).cpu().data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            answer_list = np.append(answer_list, interation['POS_ITEM_ID'].unsqueeze(1).cpu().data.numpy(), axis=0)
        i += 1

    return get_full_sort_score(epoch, answer_list, pred_list)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    recall_dict = {}
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            #sum_recall += len(act_set & pred_set) / float(len(act_set))
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
            #sum_mrr += np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            one_user_mrr = np.reciprocal(np.where(r==1)[0]+1, dtype=float)[0]
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
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
        ndcg_dict[user_id] = dcg_k / idcg
    return res / float(len(actual)), ndcg_dict


# Calculates the ideal discounted cumulative gain at k（IDCG@K）
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def get_full_sort_score(epoch, answers, pred_list):
    recall, ndcg, mrr = [], [], 0
    recall_dict_list = []
    ndcg_dict_list = []
    # for k in [1, 5, 10, 15, 20, 40]:
    for k in [1, 5, 10, 15, 20]:
        recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
        recall.append(recall_result)
        recall_dict_list.append(recall_dict_k)
        ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
        ndcg.append(ndcg_result)
        ndcg_dict_list.append(ndcg_dict_k)
    mrr, mrr_dict = cal_mrr(answers, pred_list)
    post_fix = {
        "Epoch": epoch,
        "HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]),
        "HIT@5": '{:.8f}'.format(recall[1]), "NDCG@5": '{:.8f}'.format(ndcg[1]),
        "HIT@10": '{:.8f}'.format(recall[2]), "NDCG@10": '{:.8f}'.format(ndcg[2]),
        "HIT@15": '{:.8f}'.format(recall[3]), "NDCG@15": '{:.8f}'.format(ndcg[3]),
        "HIT@20": '{:.8f}'.format(recall[4]), "NDCG@20": '{:.8f}'.format(ndcg[4]),
        # "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
        "MRR": '{:.8f}'.format(mrr)
    }
    # print(post_fix, flush=True)
    return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

