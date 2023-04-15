from torch import optim
from config import device, ACC_STEPS, NUM_BLOCKS


def train(train_dataloader, model, loss_fn, writer, epoch_ID, learning_rate=5e-4):
    i, accumulation_steps = 0, ACC_STEPS

    total_train_step = 0
    total_monitor_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0
    total_loss_3 = 0
    total_loss_4 = 0

    if epoch_ID < 15:
        learning_rate = 1e-3
    else:
        learning_rate = 1e-4

    model.to(device)
    model.train()

    for data in train_dataloader:
        i += 1
        embed, atten, coor_label, L = data

        coor_label = coor_label.to(device)
        embed = embed.to(device)
        atten = atten.to(device)

        pred_coor_ls, pred_x2d = model(embed, atten)
        # pred_coor_ls = [i.reshape(-1,3) for i in pred_coor_ls]
        # coor_label = coor_label.reshape(-1, 3)

        loss1 = loss_fn(pred_coor_ls[0], coor_label, num_block=0)
        loss2 = loss_fn(pred_coor_ls[1], coor_label, num_block=1)
        loss3 = loss_fn(pred_coor_ls[2], coor_label, num_block=2)
        loss4 = loss_fn(pred_coor_ls[3], coor_label, num_block=3)

        loss = loss1 + loss2 + loss3 + loss4

        monitor_loss = loss_fn(pred_coor_ls[3], coor_label)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss.backward()
        if ((i+1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_train_step = total_train_step + 1

        total_monitor_loss += monitor_loss.item() 
        avg_monitor_loss = total_monitor_loss / total_train_step

        total_loss_1 += loss1.item() 
        avg_loss1 = total_loss_1 / total_train_step

        total_loss_2 += loss2.item() 
        avg_loss2 = total_loss_2 / total_train_step

        total_loss_3 += loss3.item() 
        avg_loss3 = total_loss_3 / total_train_step

        total_loss_4 += loss4.item() 
        avg_loss4 = total_loss_4 / total_train_step

        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, (loss.item())/NUM_BLOCKS))
            writer.add_scalar("avg_monitor_loss", avg_monitor_loss, total_train_step)
            writer.add_scalar("avg_loss1", avg_loss1, total_train_step)
            writer.add_scalar("avg_loss2", avg_loss2, total_train_step)
            writer.add_scalar("avg_loss3", avg_loss3, total_train_step)
            writer.add_scalar("avg_loss4", avg_loss4, total_train_step)

    return avg_loss1, avg_loss2, avg_loss3, avg_loss4, avg_monitor_loss