import pdb
from logging import getLogger
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix

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
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def predicting(args, model, device, loader, epoch, stage='val'):
    model.eval()
    total_loss = 0.0
    total_auc = 0.0
    total_batches = len(loader)
    loss_bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for data in loader:
            output = model.predict(args, data, device)
            cur_loss = loss_bce(output, data[-1].float().to(device))
            auc_result = metric_info(np.array(output.cpu()), np.array(data[-1]))
            total_loss += cur_loss
            total_auc += auc_result
        average_loss = total_loss / total_batches
        average_auc = total_auc / total_batches
        if stage=='val':
            print(f'Epoch {epoch}: Valid set logloss: {average_loss}')
            print(f'Epoch {epoch}: Valid set Auc: {average_auc}')
        elif stage=='test':
            print(f' Test set logloss: {average_loss}')
            print(f' Test set Auc: {average_auc}')

    return average_loss, average_auc


def _binary_clf_curve(trues, preds):
    trues = trues == 1

    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps


def auc(x, y):
    if x.shape[0] < 2:
        raise ValueError(
            "At least 2 points are needed to compute area under curve, but x.shape = %s"
            % x.shape
        )

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing : {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area


def metric_info(preds, trues):
    fps, tps = _binary_clf_curve(trues, preds)
    if len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        logger = getLogger()
        logger.warning(
            "No negative samples in y_true, "
            "false positive value should be meaningless"
        )
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        logger = getLogger()
        logger.warning(
            "No positive samples in y_true, "
            "true positive value should be meaningless"
        )
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    result = auc(fpr, tpr)
    return result

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
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users