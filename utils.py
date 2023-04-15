# Author: Jun Li
# Function: tools for the project
# 2023-04-14

import torch
import os

from torch import nn
from config import eps
import numpy as np

def seed_torch(seed):
    """ 为整个torch环境设置随机数种子
    
    :param seed: Random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("seed: ", seed)


def model_params_transfer(model_pt_name, to_transfer_model):
    """ 从模型中加载参数，并加载到另外一个模型上
    
    :param model_pt_name: model.pt的绝对路径
    :param to_transfer_model: 传递的目标模型
    """
    model = torch.load(model_pt_name)
    model_params = nn.Module.state_dict(model)
    nn.Module.load_state_dict(to_transfer_model, model_params)


def send_email(content):
    """ 发送邮件
    
    :param content: 要写在电子邮件中的内容
    """
    import smtplib
    from email.mime.text import MIMEText
    mail_host = 'smtp.163.com'
    mail_user = 'nasainsight'
    mail_pass = 'TGKBWWXIJFQZWCGX'
    sender = 'nasainsight@163.com'
    receivers = ['nasainsight@163.com']
    message = MIMEText(content,'plain','utf-8')
    message['Subject'] = 'main.py运行结束，请查看'
    message['From'] = sender
    message['To'] = receivers[0]

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host,25)
        smtpObj.login(mail_user,mail_pass)
        smtpObj.sendmail(
            sender,receivers,message.as_string())
        smtpObj.quit()
    except smtplib.SMTPException as e:
        print('error',e) 


def weight_init(m):
    """ 将模型中Linear层、Conv2d、BatchNorm2d层的参数初始化
    
    :param m: model to initialization
    :example:
        model = nn.Linear(64,12)
        model.apply(weight_init)
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def generate_pdb(coor, pdb_name):
    """ 将coor坐标转化为PDB文件
    
    :coor: [L,3]
    :pdb_name: pdb_name
    """
    with open(pdb_name, "w") as pdb_file:
        pdb_file.write("MODEL  1\n")
        L = coor.shape[0]
        for i in range(L-1):
            x,y,z = coor[i,:]
            pdb_file.write("ATOM")
            pdb_file.write("%7d"%(i+1))
            pdb_file.write("  CA  ")
            pdb_file.write("GLY A")
            pdb_file.write("%4d"%(i+1))
            pdb_file.write("    ")
            pdb_file.write("%8.3f"%(x))
            pdb_file.write("%8.3f"%(y))
            pdb_file.write("%8.3f"%(z))
            pdb_file.write("  1.00 97.50           C  \n")
        pdb_file.write(f"TER {L}  GLY A {L} \n")
        pdb_file.write(f"ENDMDL\n")
        pdb_file.write("END\n")


def cal_r_matrix(b,c,d):
    """ 根据三个变量计算旋转矩阵
    
    :return r_matrix: [3,3]
    """
    a,b,c,d = np.array([1,b,c,d])/np.sqrt(1+b**2+c**2+d**2)
    r_matrix = np.array([a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c,
                        2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b,
                        2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]).reshape(3,3)
    return r_matrix









