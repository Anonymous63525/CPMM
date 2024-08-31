import pdb
import random

import torch
from torch.utils.data import Dataset

from SR.utils_sr import neg_sample


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.max_len = args.max_seq_length
        self.data_type = data_type

        assert self.data_type in {"train", "valid", "test"}
        if self.data_type =='train':
            for seq in user_seq:
                input_ids = seq[-(self.max_len + 2):-2]  # keeping same as train set
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
        elif self.data_type =='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])
        else:
            self.user_seq = user_seq

        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length


    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        target_pos = items[-1]


        # 为每个的训练样本选取一个负样本。从不在items的集合中随机抽样一个负样本
        target_neg = []
        seq_set = set(items)
        # target_neg.append(neg_sample(seq_set, self.args.item_size))
        target_neg = neg_sample(seq_set, self.args.item_size)

        # 填充与截断
        ori_len = len(input_ids)
        pad_len = self.max_len - len(input_ids)
        if pad_len <= 0:
            ori_len = len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]

        assert len(input_ids) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(ori_len, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )

        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(ori_len, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
            )

        return cur_tensors
