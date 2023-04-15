# Author: Jun Li
# Function: 训练并测试模型


import os
from config import name, BATCH_SIZE, EPOCH_NUM

import torch
import traceback
from loss import Coor_loss

from data.MyData import MyData, MyBatchSampler
from torch.utils.data import DataLoader
from model.CoorNet import CoorNet


train_data_path = '/home/rotation3/complex-coor-pred/data/train22310.3besm2.h5'
test_data_path = '/home/rotation3/complex-coor-pred/data/valid2000.3besm2.h5'
xyz_path = '/home/rotation3/complex-coor-pred/data/xyz.h5'
sorted_train_file = "/home/rotation3/complex-coor-pred/data/sorted_train_list.txt"
test_file = "/home/rotation3/complex-coor-pred/data/valid_list.txt"

train_ds = MyData(train_data_path, xyz_path, sorted_train_file, train_mode=False)
batch_sampler = MyBatchSampler(sorted_train_file)
trian_dl = DataLoader(train_ds, batch_sampler=batch_sampler)
test_ds = MyData(test_data_path, xyz_path, filename=test_file, train_mode=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

model = CoorNet()
l = Coor_loss()

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from scripts.train_CoorNet import train
    from scripts.test_CoorNet import test
    from _tmp.draft.init_parameters import weight_init

    # try:
    logs_folder_name = name
    epoch_num = EPOCH_NUM

    logs_name_sum_train = "logs/" + logs_folder_name + "/" + "summary/train"
    logs_name_sum_test = "logs/" + logs_folder_name + "/" + "summary/test"
    writer_sum_train = SummaryWriter(logs_name_sum_train)
    writer_sum_test = SummaryWriter(logs_name_sum_test)

    model.apply(weight_init)
    # torch.save(model, "/home/rotation3/complex-coor-pred/model/checkpoint/init.pt")

    for epoch in range(epoch_num):
        logs_name = "logs/" + logs_folder_name + "/" + "train/" + "epoch" + str(epoch)
        writer_train = SummaryWriter(logs_name)

        model_dir_name = logs_folder_name
        dir = "model/checkpoint/" + model_dir_name + "/"
        filename = "epoch" + str(epoch) + ".pt"

        if not os.path.exists(dir):
            os.mkdir(dir)
        torch.save(model, dir + filename)

        # train
        loss1, loss2, loss3, loss4, monitor_train_loss = train(trian_dl, model, l, writer_train, epoch, learning_rate=5e-4)
        writer_sum_train.add_scalars(main_tag = "epoch_train_loss",
                                     tag_scalar_dict={"loss1":loss1,
                                                      "loss2":loss2,
                                                      "loss3":loss3,
                                                      "loss4":loss4,
                                                      "monitor_loss":monitor_train_loss},
                                     global_step=epoch)

        # test  
        logs_name = "logs/" + logs_folder_name + "/" + "test/" + "epoch" + str(epoch)
        writer_test = SummaryWriter(logs_name)
        avg_test_loss = test(test_dl, model, writer_test)
        writer_sum_test.add_scalar("epoch_test_loss", avg_test_loss, epoch)

        
        # status = 0

    # except Exception as e:
    #     print("出错：{}".format(e))
    #     status = e
    #     traceback.print_exc(file=open(error_file_name, 'a'))
    #     print(traceback.format_exc())

    # from utils.send_email import send_email
    # if not status:
    #     content = "运行成功"
    #     send_email(content="运行成功")
    # else:
    #     send_email(content="运行失败\n" + str(status))
