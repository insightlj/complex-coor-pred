#实现轴向注意力中的 row Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from einops import rearrange
from config import device

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, strides=1):
        super(Residual, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.norm1 = nn.InstanceNorm2d(input_channels, affine=True)
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size, padding=self.padding, stride=strides)
        self.norm2 = nn.InstanceNorm2d(num_channels, affine=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=self.padding, stride=strides)

        use_1x1conv = False if input_channels == num_channels else True

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size, padding=self.padding, stride=strides)

        else:
            self.conv3 = None

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.conv1(self.norm1(X)))
        Y = self.conv2(self.norm2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals):
    blk = []
    for _ in range(num_residuals):
        blk.append(Residual(input_channels, num_channels))
    return blk

class ColAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, device):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.device = device
       
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels = self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels = self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels = self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.ones(1).to(self.device).requires_grad_())
        self.x2d_reduction = nn.Conv2d(in_channels=105, out_channels=64, kernel_size=3,stride=1, padding=1)   # 105是x2d embedding的维度；64是我选择的坐标系的套数


    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tuple
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''
        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.shape
       
        Q = self.query_conv(x) #size = (b,c2, h,w)
        K = self.key_conv(x)   #size = (b, c2, h, w)
        V = self.value_conv(x) #size = (b, c1,h,w)
       
        Q = Q.permute(0,3,1,2).reshape(b*w, -1,h).permute(0,2,1) #size = (b*w,h,c2)
        K = K.permute(0,3,1,2).reshape(b*w, -1,h)  #size = (b*w,c2,h)
        V = V.permute(0,3,1,2).reshape(b*w, -1,h)  #size = (b*w,c1,h)

        col_attn = torch.bmm(Q,K)

        #对row_attn进行softmax
        col_attn = self.softmax(col_attn) 
        out = torch.bmm(V,col_attn.permute(0,2,1))
       
        #size = (b,c1,h,w)
        out = out.view(b,w,-1,h).permute(0,2,3,1)
       
        out = self.gamma*out + x
 
        return out

class BiasRowAttention(nn.Module):
   
    def __init__(self, in_dim, q_k_dim, device):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(BiasRowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.device = device
       
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels = self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels = self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels = self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.ones(1).to(self.device).requires_grad_())
        self.x2d_reduction = nn.Conv2d(in_channels=105, out_channels=64, kernel_size=3, padding=1)   # 105是x2d embedding的维度；64是我选择的坐标系的套数
        self.x2d_resnet = nn.Sequential(*resnet_block(64,64,4))
        self.x2d_restore = nn.Conv2d(in_channels=64, out_channels=105, kernel_size=3, padding=1)
        

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tuple
            (x,x2d)
        x[0] : Tensor 
            4-D , (batch, in_dims, height, width) -- (b,3,64,L)
        x2d[1] : Tensor
            4-D , (batch, 105, L, L)
        '''
        x2d = x[1]
        x = x[0]
        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.shape
       
        Q = self.query_conv(x) #size = (b,c2, h,w)
        K = self.key_conv(x)   #size = (b, c2, h, w)
        V = self.value_conv(x) #size = (b, c1,h,w)
       
        Q = Q.permute(0,2,1,3).reshape(b*h, -1,w).permute(0,2,1) #size = (b*h,w,c2)
        K = K.permute(0,2,1,3).reshape(b*h, -1,w)  #size = (b*h,c2,w)
        V = V.permute(0,2,1,3).reshape(b*h, -1,w)  #size = (b*h, c1,w)

        row_attn = torch.bmm(Q,K)

        #对row_attn进行softmax
        bias = self.x2d_reduction(x2d)
        bias = self.x2d_resnet(bias)
        bias = bias.reshape(row_attn.shape)
        row_attn = row_attn + bias
        row_attn = self.softmax(row_attn)
        tmp_row_attn = rearrange(row_attn, "(b c) h w -> b c h w", b=x2d.shape[0])
        x2d = self.x2d_restore(tmp_row_attn)

        out = torch.bmm(V,row_attn.permute(0,2,1))
       
        #size = (b,c1,h,2)
        out = out.view(b,h,-1,w).permute(0,2,1,3)
        
        out = self.gamma*out + x
        return (out, x2d)


if __name__ == "__main__":
    #实现轴向注意力中的 Row Attention
    x = torch.randn(4, 3, 64,91).to(device)
    x2d = torch.randn(4,105,91,91).to(device)
    row_attn = BiasRowAttention(in_dim = 3, q_k_dim = 3,device = device).to(device)
    col_attn = ColAttention(in_dim = 3, q_k_dim=3, device=device).to(device)
    a = row_attn((x, x2d))
    b = col_attn(x)

