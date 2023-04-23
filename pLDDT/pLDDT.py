import torch
import random
from torch import nn
from model.ResNet import resnet_block
from main import train_ds, test_ds
from torch.utils.data import DataLoader
from config import device, NUM_BLOCKS
from tools.cal_lddt_multiseq import cal_lddt
from torch.utils.tensorboard import SummaryWriter
from utils import weight_init

class pLDDT(nn.Module):
    # zhang guijun老师的做法是：
    # 1. 在ResNet前面加入了轴向注意力机制
    # 2. 分别用ResNet预测出deviation和contact map, 然后据此计算pLDDT
    def __init__(self):
        super().__init__()
        self.conv_reduction = nn.Conv2d(in_channels=105, out_channels=7, kernel_size=3, stride=1, padding=1)   
                                                # 将105维的pred_x2d降维到10维，从而更好的和坐标信息融合
        self.resnet = nn.Sequential(*resnet_block(input_channels=10, num_channels=10, num_residuals=5))
        self.plddt = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self,pred_coor,pred_x2d):
        """
        pred_coor (B,L,L,3)
        pred_x2d  (B,105,L,L)  模型输出的embedding, 包含了序列embedding之后的信息
        plddt (B,L,L)
        """
        pred_x2d = self.conv_reduction(pred_x2d)
        pred_coor = torch.permute(pred_coor, (0,3,1,2))
        cmb = torch.concat((pred_coor, pred_x2d), dim=1)
        cmb = self.resnet(cmb)
        plddt = self.plddt(cmb)
        plddt = plddt.squeeze(1)
        return plddt

def train_plddt_on_a_model(net_pt_name):
    """
    function:
    从Dataloader中取出embed, atten, coor_label; 计算pred_coor与pred_x2d
    进一步计算lddt与plddt; 二者之间的L2Loss作为Loss
    
    input:
    net_pt_name 基于该模型的pred进行训练

    return:
    pred (B,L,L,3)
    pred_x2d (B,105,L,L)
    coor_label  (B,L,L,3)
    lddt (B,L,L)
    plddt (B,L,L)
    """
    global epoch
    global avg_loss
    global project_name
    global get_pLDDT
    global epoch_i

    eps = 1e-7
    local_step = 0
    global_step = 0
    total_loss = 0
    get_pLDDT = pLDDT().to(device)
    if epoch_i == 0:
        print("init model parameters……")
        get_pLDDT.apply(weight_init)    
    get_pLDDT.train()
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)   # batch_size 固定为1, 因为不同大小的蛋白质不能concat到一起形成batch
    net_pt = torch.load(net_pt_name, map_location=device)
    for param in net_pt.parameters():
        param.requires_grad = False
    opt = torch.optim.Adam(get_pLDDT.parameters(), lr=1e-3)
    train_logs = SummaryWriter("utils/plddt_logs/" + project_name +"/train/epoch" + str(epoch))

    for data in train_dataloader:
        local_step += 1
        global_step += 1
        embed, atten, coor_label, L = data
        embed = embed.to(device)
        atten = atten.to(device)
        coor_label = coor_label.to(device)
        L = L.to(device)
        pred_coor_ls, pred_x2d = net_pt(embed, atten)
        pred = pred_coor_ls[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

        predcadist = pred.unsqueeze(-2) - pred.unsqueeze(-3)   
        predcadist = ((predcadist**2).sum(dim=-1) + eps) ** 0.5   # predcadist  (N,L,L,L)
        label = ((coor_label**2).sum(dim=-1) + eps) ** 0.5
        lddt = cal_lddt(predcadist,label)
        plddt = get_pLDDT(pred, pred_x2d)

        loss = (((plddt - lddt)**2).sum() + eps) ** 0.5
        loss.backward()

        loss_item = loss.item() / 5
        total_loss = total_loss + loss_item
        avg_loss = total_loss / global_step
        
        if global_step % 100 == 0:
            train_logs.add_scalar("train_loss", avg_loss, global_step)
            print("train_step", global_step ,"train_loss:", avg_loss)

        if local_step == 4:
            opt.step()
            opt.zero_grad()
            local_step = 0

### test
def test_plddt_on_a_model(net_pt_name, plddt_epoch=11):
    """
    epoch: 0-11, 用于选择验证所训练的12个pLDDT模型
    """
    global project_name
    global get_pLDDT
    global avg_loss

    eps = 1e-7
    local_step = 0
    global_step = 0
    total_loss = 0
    avg_loss = 0
    get_pLDDT = torch.load("utils/plddt_checkpoints/" + project_name + "/epoch" + str(plddt_epoch) + ".pt")
    get_pLDDT.eval()
    # train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)   # batch_size 固定为1, 因为不同大小的蛋白质不能concat到一起形成batch
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    net_pt = torch.load(net_pt_name, map_location=device)
    for param in net_pt.parameters():
        param.requires_grad = False
    test_logs = SummaryWriter("utils/plddt_logs/" + project_name +"/test/epoch" + str(plddt_epoch))

    # print("begin to test on train_dataloader……")
    # for data in train_dataloader:
    #     local_step += 1
    #     global_step += 1
    #     embed, atten, coor_label, L = data
    #     embed = embed.to(device)
    #     atten = atten.to(device)
    #     coor_label = coor_label.to(device)
    #     L = L.to(device)
    #     pred_coor_ls, pred_x2d = net_pt(embed, atten)   # 计算pLDDT的loss的时候还需要做一遍推理……好费时间
    #     pred = pred_coor_ls[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

    #     predcadist = pred.unsqueeze(-2) - pred.unsqueeze(-3)   
    #     predcadist = ((predcadist**2).sum(dim=-1) + eps) ** 0.5   # predcadist  (N,L,L,L)
    #     label = ((coor_label**2).sum(dim=-1) + eps) ** 0.5
    #     lddt = cal_lddt(predcadist,label)
    #     plddt = get_pLDDT(pred, pred_x2d)

    #     loss = (((plddt - lddt)**2).sum() + eps) ** 0.5
    #     total_loss += loss
    #     avg_loss = total_loss / global_step
    #     test_logs.add_scalar("avg_loss_train", avg_loss, global_step)
    # print("train_dataloader avg_loss: ", avg_loss)

    print("begin to test on test_dataloader……")
    for data in test_dataloader:
        local_step += 1
        global_step += 1
        embed, atten, coor_label, L = data
        embed = embed.to(device)
        atten = atten.to(device)
        coor_label = coor_label.to(device)
        L = L.to(device)
        pred_coor_ls, pred_x2d = net_pt(embed, atten)
        pred = pred_coor_ls[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

        predcadist = pred.unsqueeze(-2) - pred.unsqueeze(-3)   
        predcadist = ((predcadist**2).sum(dim=-1) + eps) ** 0.5   # predcadist  (N,L,L,L)
        label = ((coor_label**2).sum(dim=-1) + eps) ** 0.5
        lddt = cal_lddt(predcadist,label)
        plddt = get_pLDDT(pred, pred_x2d)

        loss = (((plddt - lddt)**2).sum() + eps) ** 0.5
        total_loss += loss
        avg_loss = total_loss / global_step
        if global_step > 500:
            break
        test_logs.add_scalar("avg_loss_test", avg_loss, global_step)
    print("total_dataloader avg_loss: ", avg_loss)

if __name__ == "__main__":
    project_name = "Full_train"
    train_epoch_num = [i*3 for i in range(11)]
    random.shuffle(train_epoch_num)
    train_epoch_num += [34]
    length = len(train_epoch_num)
    ### train
    # train_epoch_logs = SummaryWriter("utils/plddt_logs/" + project_name +"/train")
    # for epoch_i in range(length):   # epoch_i 在列表中排在第i个的epoch
    #     epoch = train_epoch_num[epoch_i]
    #     print("begining to train pLDDT on model_%d" % (epoch))
    #     net_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch" + str(epoch) + ".pt"
    #     train_plddt_on_a_model(net_pt_name)
    #     train_epoch_logs.add_scalar("train_epoch_loss", avg_loss, epoch_i)
    #     if not os.path.exists("utils/plddt_checkpoints/" + project_name):
    #         os.mkdir("utils/plddt_checkpoints/" + project_name)
    #     torch.save(get_pLDDT, "utils/plddt_checkpoints/" + project_name + "/epoch" + str(epoch_i) + ".pt")
    #     print("model saved successfully!")
    #     print("*"*10 + "train over" + "*"*10 + "\n")

    print("*"*10 + "test begin" + "*"*10)
    print("begining to test pLDDT on model_34")
    net_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch" + str(34) + ".pt"
    test_epoch_loss = SummaryWriter("utils/plddt_logs/" + project_name +"/test")
    with torch.no_grad():
        for i in range(11):
            test_plddt_on_a_model(net_pt_name, plddt_epoch=i)
            test_epoch_loss.add_scalar("test_epoch_loss", avg_loss, i)

