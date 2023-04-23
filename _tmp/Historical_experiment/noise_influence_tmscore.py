import torch
from tools.cal_tm_BFGS import cal_tm, cal_tm_avg, cal_tm_max
from tools.sample_from_dataset import demo_pred_label


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
