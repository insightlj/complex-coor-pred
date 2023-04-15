# Function: 调用tmtools计算TM-score
# Author: Li Jun

import random; random.seed(20)
from tmtools import tm_align


def cal_tm(pred, label, sample_size=10):
    """ 从L套坐标系中选取sample_size套进行tm_score的计算
    
    :param pred: [L,L,3]; type:Tensor
    :param label: [L,L,3]
    :return: avg_tm
    """
    from tmtools import tm_align

    L = pred.shape[0]
    seq = "A" * int(L)
    total_tm = 0
    for _ in range(sample_size):  # 从L套序列中随机选取sample_size条序列
        index = random.randint(0, L - 1)
        pred_index, label_index = pred[index, :, :], label[index, :, :]
        pred_index, label_index = pred_index.cpu().detach().numpy(), label_index.cpu().detach().numpy()
        tm = tm_align(pred_index, label_index, seq, seq)
        total_tm += tm.tm_norm_chain1

    avg_tm = total_tm / sample_size
    return avg_tm

def cal_tm_avg(pred, label):
    """ 分别对L套坐标系进行tm_score的计算并计算平均值

    :param pred: [L,L,3]
    :param label: [L,L,3]
    """

    L = pred.shape[0]
    sample_size = L
    seq = "A" * int(L)
    total_tm = 0
    for index in range(sample_size):
        pred_index, label_index = pred[index, :,:], label[index, :,:]
        pred_index, label_index = pred_index.cpu().detach().numpy(), label_index.cpu().detach().numpy()
        tm = tm_align(pred_index, label_index, seq, seq)
        total_tm += tm.tm_norm_chain1

    avg_tm = total_tm / sample_size
    return avg_tm

def cal_tm_max(pred, label):
    """分别对L套坐标系进行tm_score的计算并取得最大值

    :param pred: [L,L,3]
    :param label: [L,L,3]
    """
    L = pred.shape[0]
    sample_size = L
    seq = "A" * int(L)
    tm_max = 0
    for index in range(sample_size):
        pred_index, label_index = pred[index, :,:], label[index, :,:]
        pred_index, label_index = pred_index.cpu().detach().numpy(), label_index.cpu().detach().numpy()
        tm = tm_align(pred_index, label_index, seq, seq)
        tm = tm.tm_norm_chain1
        if tm > tm_max:
            tm_max = tm
    return tm_max


