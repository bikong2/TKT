# coding: utf-8
# create by tongshiwei on 2019/4/12

import torch
from gluonnlp.data import FixedBucketSampler, PadSequence
from tqdm import tqdm

from TKT.shared.etl import *


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    num_buckets = params.num_buckets
    batch_size = params.batch_size

    responses = raw_data

    # 不同长度的样本每次都会分配到固定的bucket当中
    batch_idxes = FixedBucketSampler([len(rs) for rs in responses], batch_size, num_buckets=num_buckets)
    batch = []

    def index(r):
        correct = 0 if r[1] <= 0 else 1
        return r[0] * 2 + correct

    for batch_idx in tqdm(batch_idxes, "batchify"):
        batch_rs = []
        batch_pick_index = []
        batch_labels = []
        for idx in batch_idx:
            # 1. 答对答错存储的id不一样
            # 2. batch_pick_index[i], batch_labels[i]维度比batch_rs[i]小1
            batch_rs.append([index(r) for r in responses[idx]])
            if len(responses[idx]) <= 1:
                pick_index, labels = [], []
            else:
                pick_index, labels = zip(*[(r[0], 0 if r[1] <= 0 else 1) for r in responses[idx][1:]])
            batch_pick_index.append(list(pick_index))
            batch_labels.append(list(labels))
        # 每一个bucket中的max_len不一样
        max_len = max([len(rs) for rs in batch_rs])
        padder = PadSequence(max_len, pad_val=0)
        # batch_rs padding到max_len之后的数据
        # data_mask表示对应的有效数据的条数
        batch_rs, data_mask = zip(*[(padder(rs), len(rs)) for rs in batch_rs])

        max_len = max([len(rs) for rs in batch_labels])
        padder = PadSequence(max_len, pad_val=0)
        batch_labels, label_mask = zip(*[(padder(labels), len(labels)) for labels in batch_labels])
        batch_pick_index = [padder(pick_index) for pick_index in batch_pick_index]
        # Load
        # 所有数据都padding到固定长度的序列
        # data_mask label_mask表示有效数据的长度
        # len(batch_labels) + 1 = len(batch_rs)
        batch.append(
            [torch.tensor(batch_rs), torch.tensor(data_mask), torch.tensor(batch_labels),
             torch.tensor(batch_pick_index),
             torch.tensor(label_mask)])

    return batch


def pesudo_data_iter(_cfg):
    return transform(pseudo_data_generation(_cfg), _cfg)


def etl(data_src, params):
    raw_data = extract(data_src)
    print(raw_data[0])
    print(raw_data[-1])
    return transform(raw_data, params)


if __name__ == '__main__':
    """
    from longling.lib.structure import AttrDict
    import os

    filename = "/Users/xihuali/Documents/EduAI/code/TKT/data/ktbd/ednet/test.json"

    print(os.path.abspath(filename))

    for data in tqdm(extract(filename)):
        pass

    parameters = AttrDict({"batch_size": 128, "num_buckets": 100})
    for data in tqdm(etl(filename, params=parameters)):
        pass
    """
