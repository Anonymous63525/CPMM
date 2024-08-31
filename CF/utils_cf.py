import pdb
from logging import getLogger
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from scipy.sparse import csr_matrix, coo_matrix
import pandas as pd
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


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, args):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = 'user_id'
        self.uid_field = 'user_id'
        self.ITEM_ID = 'item_id'
        self.iid_field = 'item_id'
        self.NEG_ITEM_ID = 'neg_item_id'
        self.inter_feat = args.inter_feat
        self.n_users = args.num_users
        self.n_items = args.max_item
        self.inter_num = len(args.inter_feat[self.ITEM_ID])

        # load parameters info
        self.device = args.device

    def inter_matrix(self, form="coo", value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )
        return self._create_sparse_matrix(
            self.inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def _create_sparse_matrix(
        self, df_feat, source_field, target_field, form="coo", value_field=None
    ):
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if value_field is None:
            data = np.ones(len(df_feat[source_field]))
        else:
            if value_field not in df_feat:
                raise ValueError(
                    f"Value_field [{value_field}] should be one of `df_feat`'s features."
                )
            data = df_feat[value_field]

        mat = coo_matrix(
            (data, (src, tgt)), shape=(self.n_users, self.n_items)
        )

        if form == "coo":
            return mat
        elif form == "csr":
            return mat.tocsr()
        else:
            raise NotImplementedError(
                f"Sparse matrix format [{form}] has not been implemented."
            )


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

def xavier_normal_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        outputs = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2

class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)

class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> outputs = loss(pos_score, neg_score)
        >>> outputs.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

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

    def __call__(self, score, model):
        # score HIT@10 NDCG@10
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            if self.verbose:
                print(f'Validation score increased.  Saving model ...')
            torch.save(model.state_dict(), self.checkpoint_path)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Validation score increased.  Saving model ...')
            torch.save(model.state_dict(), self.checkpoint_path)
            self.counter = 0

    # def save_checkpoint(self, score, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         print(f'Validation score increased.  Saving model ...')
    #     torch.save(model.state_dict(), self.checkpoint_path)
    #     self.score_min = score

def predicting(args, model, device, rec_data_iter, epoch, stage='val'):
    model.eval()
    answer_list = None
    # for i, batch in rec_data_iter:
    i = 0
    for batch in rec_data_iter:
        interation = {
            'user_id': batch[0].to(device),  # user_id for testing,
            'item_id': batch[1].to(device),  # user_id for testing,
        }
        # 0. batch_data will be sent into the device(GPU or cpu)
        rating_pred = model.full_sort_predict(interation)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = interation['user_id'].cpu().numpy()
        rating_pred[args.train_matrix[batch_user_index].toarray() > 0] = 0
        ind = np.argpartition(rating_pred, -40)[:, -40:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if i == 0:
            pred_list = batch_pred_list
            answer_list = interation['item_id'].unsqueeze(1).cpu().data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            answer_list = np.append(answer_list, interation['item_id'].unsqueeze(1).cpu().data.numpy(), axis=0)
        i += 1

    return get_full_sort_score(epoch, answer_list, pred_list, stage)

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

def get_full_sort_score(epoch, answers, pred_list, stage):
    recall, ndcg, mrr = [], [], 0
    recall_dict_list = []
    ndcg_dict_list = []
    for k in [1, 5, 10, 15, 20, 40]:
        recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
        recall.append(recall_result)
        recall_dict_list.append(recall_dict_k)
        ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
        ndcg.append(ndcg_result)
        ndcg_dict_list.append(ndcg_dict_k)
    mrr, mrr_dict = cal_mrr(answers, pred_list)
    post_fix = {
        "Stage": stage,
        "Epoch": epoch,
        "HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]),
        "HIT@5": '{:.8f}'.format(recall[1]), "NDCG@5": '{:.8f}'.format(ndcg[1]),
        "HIT@10": '{:.8f}'.format(recall[2]), "NDCG@10": '{:.8f}'.format(ndcg[2]),
        "HIT@15": '{:.8f}'.format(recall[3]), "NDCG@15": '{:.8f}'.format(ndcg[3]),
        "HIT@20": '{:.8f}'.format(recall[4]), "NDCG@20": '{:.8f}'.format(ndcg[4]),
        "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
        "MRR": '{:.8f}'.format(mrr)
    }
    print(post_fix, flush=True)
    return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4],
            recall[5], ndcg[5], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

def generate_rating_matrix_valid(user_seq, item_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = user_seq
    col = item_seq
    data = np.ones_like(col)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # 创建一个基于压缩稀疏行（Compressed Sparse Row, CSR）格式的评分矩阵（rating matrix）的操作
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def get_user_seqs(data_file):
    lines = pd.read_csv(data_file, index_col=None)
    user_seq = []
    item_seq = []
    for index, line in lines.iterrows():
        user_seq.append(line['user_id'])
        item_seq.append(line['item_id'])

    max_item = max(item_seq)
    max_user = max(user_seq)

    num_items = max_item + 1
    num_users = max_user + 1

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, item_seq, num_users, num_items)
    return valid_rating_matrix, num_users, num_items

def xavier_uniform_initialization(module):

    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)