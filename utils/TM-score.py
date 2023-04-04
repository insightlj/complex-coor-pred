import torch
from utils.demo_of_model_and_eval import demo_pred_label
import random

random.seed(20)

def cal_tm(pred, label, sample_size=10):
    """
    pred [L,L,3]
    label [L,L,3]
    sample_size 从L套坐标系中选取sample_size套进行tm_score的计算
    """
    from tmtools import tm_align

    L = pred.shape[0]
    seq = "A" * int(L)
    total_tm = 0
    for _ in range(sample_size):
        index = random.randint(0, L - 1)
        pred_index, label_index = pred[index, :, :], label[index, :, :]
        pred_index, label_index = pred_index.cpu().detach().numpy(), label_index.cpu().detach().numpy()
        tm = tm_align(pred_index, label_index, seq, seq)
        total_tm += tm.tm_norm_chain1

    avg_tm = total_tm / sample_size
    return avg_tm

def cal_tm_avg(pred, label):
    """分别对L套坐标系进行tm_score的计算并计算平均值
    pred [L,L,3]
    label [L,L,3]
    """
    from tmtools import tm_align

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

from tmtools import tm_align
def cal_tm_max(pred, label):
    """分别对L套坐标系进行tm_score的计算并取得最大值
    pred [L,L,3]
    label [L,L,3]
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


# ---------------------------------if __name__ == "__main__"-------------------------------------------------------------
# cal max_tm
if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= "3"
    
    import time
    beg = time.time()
    eval_size = 1
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"
    pred, label = demo_pred_label(demo_size=eval_size, net_pt_name=model_pt_name, shuffle=True)
    pred.squeeze_()
    label.squeeze_()
    tm_avg = cal_tm(pred, label)
    tm_max = cal_tm_max(pred, label)
    print(tm_avg, tm_max)
    t = time.time() - beg
    print("tm_cal time use:", t)
    print(label.shape[-2])


# cal tm-score
if __name__ == "__ main__":
    eval_size = 10  # 用来评估模型效果的Batch的数量
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"

    pred_ls, label_ls = demo_pred_label(eval_size, model_pt_name)
    demo_tm = 0
    for i in range(eval_size):
        pred = pred_ls[i]
        label = label_ls[i]
        tm = cal_tm(pred, label, sample_size=pred.shape[0])
        print("sample{}: tm-score:{}".format(i + 1, tm))
        demo_tm += tm
    demo_tm = demo_tm / eval_size
    print("avg_tm_score:{}".format(demo_tm))

# cal traverse tm-score
if __name__ == "__ main__":
    eval_size = 10  # 用来评估模型效果的Batch的数量
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"

    pred_ls, label_ls = demo_pred_label(eval_size, model_pt_name)
    demo_tm = 0
    for i in range(eval_size):
        pred = pred_ls[i]
        label = label_ls[i]
        tm = cal_tm(pred, label)
        print("sample{}: tm-score:{}".format(i + 1, tm))
        demo_tm += tm
    demo_tm = demo_tm / eval_size
    print("avg_tm_score:{}".format(demo_tm))



# cal tm-score and save specific pred
if __name__ == "__ main__":
    eval_size = 10  # 用来评估模型效果的Batch的数量
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch30.pt"

    pred_ls, label_ls = demo_pred_label(eval_size, model_pt_name)
    demo_tm = 0
    for i in range(eval_size):
        pred = pred_ls[i]
        label = label_ls[i]
        tm = cal_tm(pred, label)
        print("sample{}: tm-score:{}".format(i + 1, tm))
        demo_tm += tm
        if tm > 0.5 and tm < 0.6:
            torch.save(pred, "/home/rotation3/complex-coor-pred/data/notBadProtein/pred" + str(i) + ".npy")
            torch.save(label, "/home/rotation3/complex-coor-pred/data/notBadProtein/label" + str(i) + ".npy")
            print("saved!")
    demo_tm = demo_tm / eval_size
    print("avg_tm_score:{}".format(demo_tm))

# cal noise's influence to tm-score
if __name__ == "__ main__":
    eval_size = 20   # 用来评估模型效果的Batch的数量
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch14.pt"

    pred_ls, label_ls = demo_pred_label(eval_size, model_pt_name)
    demo_tm = 0
    not_bad_tm_num = 0
    demo_tm_noise = 0
    lager_num = 0
    for i in range(eval_size):
        pred = pred_ls[i]
        label = label_ls[i]
        pred_noise = pred * (torch.ones_like(pred, device=pred.device) + 0.01*torch.randn(pred.shape, device=pred.device))
        tm = cal_tm(pred, label)
        tm_noise =cal_tm(pred_noise, label)
        print("sample{}: tm-score:{}".format(i+1, tm))
        print("sample{}_noise: tm-score:{}".format(i+1, tm_noise))
        if tm > 0.5:
            not_bad_tm_num += 1
            if tm_noise > tm:
                print("sample_noise is larger in not bad result:{}".format(tm_noise - tm))
                lager_num += 1
        print("---------")
        demo_tm += tm
        demo_tm_noise += tm_noise
    demo_tm = demo_tm / eval_size
    demo_tm_noise = demo_tm_noise / eval_size
    print("avg_tm_score:{}".format(demo_tm))
    print("avg_tm_score_noise:{}".format(demo_tm_noise))
    print("precent of improvement among not bad pred result: " + str(lager_num / not_bad_tm_num))

# cal a series of noise's influence to given pred result
if __name__ == "__ main__":
    pred = torch.load("/home/rotation3/complex-coor-pred/data/badProteins/pred2.npy")
    label = torch.load("/home/rotation3/complex-coor-pred/data/badProteins/label2.npy")

    demo_tm_noise = 0
    lager_num = 0
    eval_size = 0
    better_tm_score = 0
    noise_step = 10
    noise_ls = []

    tm = cal_tm(pred, label)

    for i in range(noise_step):
        pred_noise = pred * (torch.ones_like(pred, device=pred.device) + 0.02*torch.randn(pred.shape, device=pred.device))
        tm_noise =cal_tm(pred_noise, label)
        noise_ls.append(tm_noise)
        print("tm-score:{}".format(tm_noise))

        if tm_noise - tm > 0:
            print("!!!!sample_noise is larger:{}".format(tm_noise - tm))
            torch.save(pred_noise, "/home/rotation3/complex-coor-pred/data/notBadProtein/better_pred/pred_noise" + str(i) + ".npy")
            lager_num += 1
            better_tm_score += tm_noise
        print("---------")
        demo_tm_noise += tm_noise
        eval_size += 1

    avg_demo_tm_noise = demo_tm_noise / eval_size
    print("normal_tm_score:{}".format(tm))
    print("avg_tm_score_noise:{}".format(demo_tm_noise / eval_size))
    print("better_tm_score:{}".format(better_tm_score / (lager_num + 1e-4)))
    print("precent" + str(lager_num / eval_size))
