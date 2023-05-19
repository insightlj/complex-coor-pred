import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def f(array, name, bin=100):
    plt.figure(figsize=(12, 8))  #设置画布的大小
    sns.set_palette("hls")       #设置所有图的颜色，使用hls色彩空间
    sns.distplot(np.array(array),color="steelblue",bins=bin,kde=True)
    plt.xlabel('LDDT',fontsize=20)           #添加x轴标签，并改变字体
    plt.ylabel('times',fontsize=20)   #添加y轴变浅，并改变字体
    plt.grid(linestyle='-')   #添加网格线
    plt.xticks(fontsize=15)   #改变x轴字体大小
    plt.yticks(fontsize=15)   #改变y轴字体大小
    sns.despine(ax=None, top=True, right=True, left=True,bottom=True)    #将图像的框框删掉
    plt.savefig(name + ".png")   #保存图片

b = np.load('scripts/plot_data/CoorNet_lddt.npy')
c = np.load('scripts/plot_data/ResNet_lddt.npy')

# f(b, 'CoorNet_lddt_train')
# f(c, 'ResNet_lddt')

fig = plt.figure(figsize=(12, 8))  #设置画布的大小
# plt.xlim(0,1)
sns.set_palette("hls")       #设置所有图的颜色，使用hls色彩空间
sns.distplot(np.array(b),color="steelblue",bins=100,kde=True,label='Multi-Block')
sns.distplot(np.array(c),bins=100,kde=True,label='Single-Block')
plt.xlabel('LDDT',fontsize=20)           #添加x轴标签，并改变字体
plt.ylabel('times',fontsize=20)   #添加y轴变浅，并改变字体
plt.grid(linestyle='-')   #添加网格线
plt.xticks(fontsize=15)   #改变x轴字体大小
plt.yticks(fontsize=15)   #改变y轴字体大小
sns.despine(ax=None, top=True, right=True, left=True,bottom=True)    #将图像的框框删掉
fig.legend(labels=["Multi-Block", "Single-Block"])
plt.savefig('bc' + ".png")   #保存图片