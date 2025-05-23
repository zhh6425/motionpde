
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import math
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils import *
from models import *

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#B*C*H*W->B*C*1*1.....->B*C*1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # print(in_planes,in_planes // 2)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # print('x:',x.shape)
        # print('self.avg_pool(x):',self.avg_pool(x).shape)
        # print('elf.fc1(self.avg_pool(x)):',self.fc1(self.avg_pool(x)).shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print('avg_out:',avg_out.shape)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


def group_points_4DV_T_S(points, args):
    #B*F*512*4
    T_ball_radius = torch.tensor(0.06)
    cur_train_size = points.shape[0]#
    INPUT_FEATURE_NUM = points.shape[-1]#3
    
    points = points.view(cur_train_size*args.framenum, args.EACH_FRAME_SAMPLE_NUM, -1)#(B*F)*512*4
    # print('1points:',points.shape,cur_train_size,args.framenum, args.EACH_FRAME_SAMPLE_NUM, -1)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size*args.framenum,args.T_sample_num_level1,3,args.EACH_FRAME_SAMPLE_NUM) \
                 - points[:,0:args.T_sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size*args.framenum,args.T_sample_num_level1,3,args.EACH_FRAME_SAMPLE_NUM)# (B*F )* 64 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024 distance
    dists, inputs1_idx = torch.topk(inputs1_diff, args.T_knn_K, 2, largest=False, sorted=False)  # dists: B * 512 * 32; inputs1_idx: B * 512 * 32
    
    # ball query
    invalid_map = dists.gt(T_ball_radius) # B * 512 * 64  value: binary
    
    for jj in range(args.T_sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size*args.framenum,args.T_sample_num_level1*args.T_knn_K,1).expand(cur_train_size*args.framenum,args.T_sample_num_level1*args.T_knn_K,INPUT_FEATURE_NUM)

    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size*args.framenum,args.T_sample_num_level1,args.T_knn_K,INPUT_FEATURE_NUM) # (B*F)*64*32*4

    inputs_level1_center = points[:,0:args.T_sample_num_level1,0:INPUT_FEATURE_NUM ].unsqueeze(2)       # (B*F)*64*1*4
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center[:,:,:,0:3].expand(cur_train_size*args.framenum,args.T_sample_num_level1,args.T_knn_K,3)# (B*F)*64*32*3
    if(1==1):
        dis_l=torch.mul(inputs_level1[:,:,:,0:3], inputs_level1[:,:,:,0:3])
        
        dis_l=dis_l.sum(3).unsqueeze(3)#lx#
        
        inputs_level1 = torch.cat((inputs_level1,dis_l),3).unsqueeze(1).transpose(1,4).squeeze(4)  # (B*F)*4*64*32
    
    inputs_level1_center = inputs_level1_center.contiguous().view(cur_train_size,args.framenum,args.T_sample_num_level1,1,INPUT_FEATURE_NUM).transpose(2,3).transpose(2,4)   # (B*F)*4*64*1
    FEATURE_NUM = inputs_level1.shape[-3]#4
    inputs_level1=inputs_level1.view(cur_train_size,args.framenum,FEATURE_NUM, args.T_sample_num_level1, args.T_knn_K)#B*F*4*Cen*K
    return inputs_level1, inputs_level1_center


def group_points_4DV_T_S2(points, args):
    #B*F*Cen1*(3+128)
    T_ball_radius = torch.tensor(0.11)
    cur_train_size = points.shape[0]#
    INPUT_FEATURE_NUM = points.shape[-1]#4
    
    points = points.view(cur_train_size*args.framenum, args.T_sample_num_level1, -1)#(B*F)*512*4
    # print('1points:',points.shape,cur_train_size,args.framenum, args.T_sample_num_level1, -1)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size*args.framenum,args.T_sample_num_level2,3,args.T_sample_num_level1) \
                 - points[:,0:args.T_sample_num_level2,0:3].unsqueeze(-1).expand(cur_train_size*args.framenum,args.T_sample_num_level2,3,args.T_sample_num_level1)# (B*F )* 64 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024 distance
    # print('inputs1_diff:',inputs1_diff.shape)
    dists, inputs1_idx = torch.topk(inputs1_diff, args.T_knn_K2, 2, largest=False, sorted=False)  # dists: B * 512 * 32; inputs1_idx: B * 512 * 32
    # print('inputs1_idx:',inputs1_idx.shape)
    # ball query
    invalid_map = dists.gt(T_ball_radius) # B * 512 * 64  value: binary
    
    for jj in range(args.T_sample_num_level2):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size*args.framenum,args.T_sample_num_level2*args.T_knn_K2,1).expand(cur_train_size*args.framenum,args.T_sample_num_level2*args.T_knn_K2,points.shape[-1])
    # print('points:',points.shape)
    # print('pointsg:',points.gather(1,idx_group_l1_long).shape)
    # print(cur_train_size*args.framenum,args.T_sample_num_level2,args.T_knn_K2,points.shape[-1])
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size*args.framenum,args.T_sample_num_level2,args.T_knn_K2,points.shape[-1]) # (B*F)*64*32*4

    inputs_level1_center = points[:,0:args.T_sample_num_level2,0:args.INPUT_FEATURE_NUM].unsqueeze(2)       # (B*F)*64*1*4
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center[:,:,:,0:3].expand(cur_train_size*args.framenum,args.T_sample_num_level2,args.T_knn_K2,3)# (B*F)*64*32*3
    if(1==1):
        dis_l=torch.mul(inputs_level1[:,:,:,0:3], inputs_level1[:,:,:,0:3])
        
        dis_l=dis_l.sum(3).unsqueeze(3)#lx#
        
        inputs_level1 = torch.cat((inputs_level1,dis_l),3).unsqueeze(1).transpose(1,4).squeeze(4)  # (B*F)*4*C2en*32
    
    inputs_level1_center = inputs_level1_center.contiguous().view(cur_train_size,args.framenum,args.T_sample_num_level2,1,args.INPUT_FEATURE_NUM).transpose(2,3).transpose(2,4)   # (B*F)*4*64*1
    FEATURE_NUM = inputs_level1.shape[-3]#4
    inputs_level1=inputs_level1.view(cur_train_size,args.framenum,FEATURE_NUM, args.T_sample_num_level2, args.T_knn_K2)#B*F*4*Cen*K
    return inputs_level1, inputs_level1_center


def get_positional_encoding(max_seq_len, embed_dim):

    positional_encoding=torch.zeros(max_seq_len, embed_dim).cuda()
    for pos in range(max_seq_len):
        for i in range(embed_dim):
            if(i%2==0):
                positional_encoding[pos,i]=torch.sin(pos / torch.tensor(10000**(2 * i / embed_dim)))
            else:
                positional_encoding[pos,i]=torch.cos(pos / torch.tensor(10000**(2 * i /embed_dim)))
    return positional_encoding

nstates_plus_1 = [64,64,128]
nstates_plus_2 = [128,128,256]
nstates_plus_3 = [256,512,1024,1024,256]

S_nstates_plus_1 = [64,64,128]
S_nstates_plus_2 = [128,128,256]
T_nstates_plus_2 = [256,512,1024]
T_nstates_plus_3 = [1024]
vlad_dim_out = 128*8
dim_out=1024

@MODEL_REGISTRY.register()
class SequentialPointNet(nn.Module):
    def __init__(self, args, num_classes):
        super(SequentialPointNet, self).__init__()
        self.backbone_dim = 1024
        self.temperal_num = args.temperal_num
        self.knn_K = args.knn_K
        self.ball_radius2 = args.ball_radius2
        self.sample_num_level1 = args.sample_num_level1
        self.sample_num_level2 = args.sample_num_level2
        self.INPUT_FEATURE_NUM = args.INPUT_FEATURE_NUM # x,y,x,c : 4
        # self.num_outputs = args.Num_Class
        ####SAMPLE_NUM
        self.Seg_size = args.Seg_size
        self.stride=args.stride
        self.EACH_FRAME_SAMPLE_NUM=args.EACH_FRAME_SAMPLE_NUM
        self.T_knn_K = args.T_knn_K
        self.T_knn_K2= args.T_knn_K2
        self.T_sample_num_level1 = args.T_sample_num_level1
        self.T_sample_num_level2 = args.T_sample_num_level2
        self.framenum=args.framenum
        self.T_group_num=int((self.framenum-self.Seg_size)/self.stride)+1

        self.args=args
        self.dim=128

        self.normalize_input=True
        self.pooling = args.pooling

        self.netR_T_S1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM+1, S_nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_1[0], S_nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_1[1], S_nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.T_knn_K),stride=1)
            )
        self.ca_S2 = ChannelAttention(self.INPUT_FEATURE_NUM+1+S_nstates_plus_1[2])
        self.netR_T_S2 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM+1+S_nstates_plus_1[2], S_nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_2[0], S_nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_2[1], S_nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.T_knn_K2),stride=1)
            )
        self.ca_T1 = ChannelAttention(self.INPUT_FEATURE_NUM+S_nstates_plus_2[2])
        self.net4DV_T1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K(B*10*28*2048)
            nn.Conv2d(self.INPUT_FEATURE_NUM+S_nstates_plus_2[2], T_nstates_plus_2[0], kernel_size=(1, 1)),#10->64
            nn.BatchNorm2d(T_nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(T_nstates_plus_2[0], T_nstates_plus_2[1], kernel_size=(1, 1)),#64->64
            nn.BatchNorm2d(T_nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(T_nstates_plus_2[1], T_nstates_plus_2[2], kernel_size=(1, 1)),#64->128
            nn.BatchNorm2d(T_nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.T_sample_num_level2*self.Seg_size),stride=1)#1*（t*512）#B*C*G*1
            )
        self.net4DV_T2 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(T_nstates_plus_2[2], T_nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(T_nstates_plus_3[0]),
            nn.ReLU(inplace=True),
        )

        KerStr=[(24,24),(12,12)]
        self.maxpoolings = nn.ModuleList([nn.MaxPool2d((K[0],1 ),(K[1],1)) for K in KerStr])
        self.PE=get_positional_encoding(self.framenum,T_nstates_plus_2[2])

        # self.netR_FC = nn.Sequential(
        #     # B*1024
        #     nn.Linear(dim_out*4+256, nstates_plus_3[4]),
        #     nn.BatchNorm1d(nstates_plus_3[4]),
        #     nn.ReLU(inplace=True),
        #     # B*512
        #     nn.Linear(nstates_plus_3[4], self.num_outputs),
        #     nn.BatchNorm1d(self.num_outputs),
        #     nn.ReLU(inplace=True),
        #     # B*num_outputs
        # )

    def forward(self, clips_input):
        end_points = {}
        xt, yt = group_points_4DV_T_S(clips_input, self.args)#B*F*4*Cen*K  B*F*4*Cen*1

        B,f,d,N,k = xt.shape#B*F*4*Cen*K
        yt=yt.view(B*f,yt.size(2), self.args.T_sample_num_level1, 1)#(B*F)*4*Cen*1
        xt=xt.view(B*f,d, self.args.T_sample_num_level1, k)#(B*F)*4+1*Cen*K
        xt = self.netR_T_S1(xt)#(B*F)*128*Cen*1
        xt = torch.cat((yt, xt),1).squeeze(-1)#(B*F)*(4+128)*Cen
        xt=xt.view(B,f,xt.size(1), self.args.T_sample_num_level1).transpose(2,3)#(B*F)*(4+128)*Cen->B*F*Cen1*(4+128)
        S_inputs_level2,inputs_level1_center_s2 =group_points_4DV_T_S2(xt,self.args)##B*F*5+128*Cen2*K2   B*F*4*Cen2*1
        B2,f2,d2,N2,k2 = S_inputs_level2.shape#B*F*4*Cen*K
        inputs_level1_center_s2=inputs_level1_center_s2.view(B2*f2,inputs_level1_center_s2.size(2), self.args.T_sample_num_level2, 1)#(B*F)*4*C2en*1
        S_inputs_level2=S_inputs_level2.view(B2*f2,d2, self.args.T_sample_num_level2, k2)#(B*F)*5+128*C2en*K2
        S_inputs_level2 = self.ca_S2(S_inputs_level2) * S_inputs_level2
        xt = self.netR_T_S2(S_inputs_level2)#(B*F)*128*Cen2*1
        
        ###res s2
        xt_resS2=xt.squeeze(-1).view(B,f,xt.size(1), self.args.T_sample_num_level2).transpose(1,2)#B*256*F*Cen2
        xt_resS2=F.max_pool2d(xt_resS2,kernel_size=(f,self.args.T_sample_num_level2)).squeeze(-1).squeeze(-1)#B*256
        
        xt = torch.cat((inputs_level1_center_s2, xt),1).squeeze(-1)#(B*F)*4+128*Cen2
        xt =xt.view(-1,self.framenum,xt.size(1),self.args.T_sample_num_level2).transpose(2,3)##(B*F)*(4+128)*C2en-》B*F*(4+128)*C2en->B*F*C2en*(4+128)
        
        T_inputs_level2 =xt.transpose(1,3).transpose(2,3)
        T_inputs_level2 = self.ca_T1(T_inputs_level2) * T_inputs_level2
        xt = self.net4DV_T1(T_inputs_level2)# B, C, T, 1
        
        ###resT1
        xt_resT1=F.max_pool2d(xt,kernel_size=(f,1)).squeeze(-1).squeeze(-1)#B*256
        xt=xt.squeeze(-1) + self.PE.transpose(0,1)
        xt=xt.unsqueeze(-1)
        xt = self.net4DV_T2(xt)# B, C, T, 1

        xt = [maxpooling(xt) for maxpooling in self.maxpoolings]#B*(2048)*[G]*1
        xt = torch.cat(xt,2) # B, C, T', 1

        # xt = xt.contiguous().view(xt.size(0),-1)
        # xt = torch.cat((xt, xt_resS2, xt_resT1), 1)
        # x = self.netR_FC(xt)
        
        return None, xt, [xt_resS2, xt_resT1]
