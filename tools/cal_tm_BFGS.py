# Function:
    # 区别于TM-score.py, cal_tm_score.py可以并行地计算多条序列下的tm-score，
    # 速度更快，并且不需要对于的序列比对的步骤
# Author: Jun Li

import numpy as np
import scipy.optimize as opt


def cal_r_matrix(b,c,d):
    a,b,c,d = np.array([1,b,c,d])/np.sqrt(1+b**2+c**2+d**2)
    r_matrix = np.array([a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c,
                        2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b,
                        2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]).reshape(3,3)
    return r_matrix

def TMscore(pred, label):
    """
    pred [L,3]
    label [L,3]
    
    return
    tmscore: item
    """
    L = pred.shape[0]
    d0 = 1.24*((L-15) ** (1 / 3)) - 1.8
    d0_square = d0 **2
    def minize_L2Loss(x):
        b,c,d = x[0], x[1], x[2]
        r_matrix = cal_r_matrix(b,c,d)
        L2Loss = ((pred - label @ r_matrix - (x[3],x[4],x[5])) ** 2 ).mean()
        return L2Loss


    res = opt.minimize(minize_L2Loss, (0,0,0,0,0,0), method = 'BFGS')
    x = res.x
    print(x)
    tmp = ((pred - label @ cal_r_matrix(x[0],x[1],x[2]) - (x[3],x[4],x[5])) ** 2 ).mean(axis=1)
    tmscore = (1/(tmp/d0_square + 1)).mean(axis=0)

    if tmscore < 0.1 or x[0]>5:
        print("BFGS失效, 尝试Powell")
        bounds = np.array([[-1,1],[-1,1],[-1,1],[None,None],[None,None],[None,None]])
        res = opt.minimize(minize_L2Loss, (0,0,0,0,0,0), method = 'Powell',bounds=bounds)
        x = res.x
        print(x)
        tmp = ((pred - label @ cal_r_matrix(x[0],x[1],x[2]) - (x[3],x[4],x[5])) ** 2 ).mean(axis=1)
        tmscore = (1/(tmp/d0_square + 1)).mean(axis=0)

        if tmscore < 0.1:
            raise RuntimeError("BFGS, Powell均失效")

    return tmscore


### 暂时还有BUG!!!不要使用, 还没有把旋转函数加进去
def TMscore_traverse(pred, label):
    """
    pred [L,L,3]
    label [L,L,3]   
    return tm_list
    """
    L = pred.shape[0]
    d0 = 1.24 * ((L-15) ** (1 / 3)) - 1.8
    d0_square = d0 **2
    def opt_tmscore(x):
        x = x.reshape((L,1,3))
        return -(1 / (((pred - label - x) ** 2).mean(axis=2) / d0_square + 1 )).mean(axis=1).mean()

    def TMscore(x):
        x = x.reshape((L,1,3))
        return (1 / (((pred - label - x) ** 2).mean(axis=2) / d0_square + 1 )).mean(axis=1)
    
    res = opt.minimize(opt_tmscore, np.zeros(L*3), method = "Powell")
    tmscore_array = TMscore(res.x)
    return tmscore_array
    

if __name__ == '__main__':
    from tools.sample_from_dataset import demo_pred_label
    eval_size = 1
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"
    pred, label = demo_pred_label(eval_size, model_pt_name, shuffle=False)
    pred = ((pred.squeeze())[:,:,:]).cpu().numpy()
    label = ((label.squeeze())[:,:,:]).cpu().numpy()

    # import time
    # # 测试哪个方法的计算最准确; 最高效
    # for method in ["Nelder-Mead", "Powell", "CG", "BFGS", 
    #                "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]:
    #     beg = time.time()
    #     res = opt.minimize(opt_tmscore, np.zeros(L*3), method = method)
    #     tmscore = TMscore(res.x)
    #     print("method: ", method)
    #     print("time: ", time.time() - beg)
    #     print(tmscore.mean())
    #     print(tmscore.max())

    ### 成功啦！！！！！为什么TMscore的分数这么低？是因为如果以最开始或者最后面的几个原子为坐标的话，是不行滴！！！
