import torch
from main import test_ds
from torch.utils.data import DataLoader
from config import NUM_BLOCKS,device


# 提取pred和label，形成数组或者列表
def demo_pred_label(demo_size, net_pt_name, shuffle=False,batch_size=1, 
                    include_x2d=False):
    """
    function:
    从Dataloader中取出embed, atten, coor_label; 
    然后通过网络计算出pred, 将计算结果写入列表中, 返回pred_ls, label_ls
    列表的大小由demo_size决定
    
    input:
    demo_size  当demo_size==1时, 会返回一个列表, 其中包含了pred和label
    shuffle   该参数作用在DataLoader上, 决定是否从中随机取数据
    include_x2d  预测结果是否包含pred_x2d信息

    return:
    pred (B,L,L,3) if include_x2d=Fasle else (pred, pred_x2d) ((L,L,3),(B,105,L,L)
    pred_x2d (B,105,L,L)
    coor_label  (B,L,L,3)
    """
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=shuffle)
    net_pt = torch.load(net_pt_name, map_location=device)

    i = 0
    ls_pred = []
    ls_label = []
    with torch.no_grad():
        for data in test_dataloader:
            i += 1
            if i > demo_size:
                break
            embed, atten, coor_label, L = data
            embed = embed.to(device)
            atten = atten.to(device)
            coor_label = coor_label.to(device)
            L = L.to(device)
            pred_coor_ls, pred_x2d = net_pt(embed, atten)
            pred = pred_coor_ls[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor
            if include_x2d:    # 如果include_x2d=True, pred_coor与pred_x2d将以元组的形式返回
                pred = (pred, pred_x2d)

            ls_pred.append(pred)
            ls_label.append(coor_label)

    if demo_size == 1:
        return (ls_pred[0], ls_label[0])
    else:
        return [ls_pred, ls_label]


# 测试代码
if __name__ == "__main__":
    DEMO_SIZE = 2
    net_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch14.pt"

    ls_pred, ls_label = demo_pred_label(demo_size = DEMO_SIZE, net_pt_name = net_pt_name)

    for i in range(DEMO_SIZE):
        print(ls_pred[i].shape, ls_label[i].shape)

    from config import loss_fn as l
    p = ls_pred[0]
    q = ls_label[0]

    print(p.shape, q.shape)
    loss1 = l(p, q, num_block=0)
    print(loss1)

    loss2 = l(p, q, num_block=1)
    print(loss2)

    loss3 = l(p, q, num_block=2)
    print(loss3)

    loss4 = l(p, q, num_block=3)
    print(loss4)

    monitor_loss = l(p, q)
    print(monitor_loss)
