import torch
from torch import nn, einsum
import sys 
import os
from einops import rearrange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils import *
from models import *
from typing import List

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, xyzs, features):
        b, l, n, _, h = *features.shape, self.heads

        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h = h), qkv)                             # [b, h, m, d]

        xyzs_flatten = rearrange(xyzs, 'b l n d -> b (l n) d')                                                      # [b, m, 3]

        delta_xyzs = torch.unsqueeze(input=xyzs_flatten, dim=1) - torch.unsqueeze(input=xyzs_flatten, dim=2)        # [b, m, m, 3]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                             # [b, h, m, m]
        attn = dots.softmax(dim=-1)

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                          # [b, h, m, d]

        attn = torch.unsqueeze(input=attn, dim=4)                                                                   # [b, h, m, m, 1]
        delta_xyzs = torch.unsqueeze(input=delta_xyzs, dim=1)                                                       # [b, 1, m, m, 3]
        delta_xyzs = torch.sum(input=attn*delta_xyzs, dim=3, keepdim=False)                                         # [b, h, m, 3]

        displacement_features = self.spatial_op(delta_xyzs)                                                         # [b, h, m, d]

        out = v + displacement_features
        out = rearrange(out, 'b h m d -> b m (h d)')
        out =  self.to_out(out)
        out = rearrange(out, 'b (l n) d -> b l n d', l=l, n=n)
        return out + features


class Transformer(nn.Module):
    def __init__(self, dim, depths, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depths):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features


class P4DConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = 'addition',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 bias: bool = False):

        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)


    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        device = xyzs.get_device()

        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):                 # temporal anchor frames
            # spatial anchor point subsampling by FPS
            anchor_idx = furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            new_feature = []
            for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                neighbor_xyz = xyzs[i]

                idx = ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                displacement = self.conv_d(displacement)

                if self.in_planes != 0:
                    neighbor_feature_grouped = grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                    feature = self.conv_f(neighbor_feature_grouped)
                    if self.operator == '+':
                        feature = feature + displacement
                    else:
                        feature = feature * displacement
                else:
                    feature = displacement

                feature = self.mlp(feature)
                if self.spatial_pooling == 'max':
                    feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                elif self.spatial_pooling == 'sum':
                    feature = torch.sum(input=feature, dim=-1, keepdim=False)
                else:
                    feature = torch.mean(input=feature, dim=-1, keepdim=False)

                new_feature.append(feature)
            new_feature = torch.stack(tensors=new_feature, dim=1)
            if self.temporal_pooling == 'max':
                new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
            elif self.temporal_pooling == 'sum':
                new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
            else:
                new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
            new_xyzs.append(anchor_xyz)
            new_features.append(new_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features


class P4DTransConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 original_planes: int = 0,
                 bias: bool = False):
        """
        Args:
            in_planes: C'. when point features are not available, in_planes is 0.
            out_planes: C"
            original_planes: skip connection from original points. when original point features are not available, original_in_planes is 0.
            bias: whether to use bias
            batch_norm: whether to use batch norm
            activation:
        """
        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm

        conv = []
        for i in range(len(mlp_planes)):
            if i == 0:
                conv.append(nn.Conv1d(in_channels=in_planes+original_planes, out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            else:
                conv.append(nn.Conv1d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                conv.append(nn.BatchNorm1d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, xyzs: torch.Tensor, original_xyzs: torch.Tensor, features: torch.Tensor, original_features: torch.Tensor = None) -> torch.Tensor:
        r"""
        Parameters
        ----------
        xyzs : torch.Tensor
            (B, T, N', 3) tensor of the xyz positions of the convolved features
        original_xyzs : torch.Tensor
            (B, T, N, 3) tensor of the xyz positions of the original points
        features : torch.Tensor
            (B, T, C', N') tensor of the features to be propigated to
        original_features : torch.Tensor
            (B, T, C, N) tensor of original point features for skip connection

        Returns
        -------
        new_features : torch.Tensor
            (B, T, C", N) tensor of the features of the unknown features
        """

        T = xyzs.size(1)

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
        features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        new_xyzs = original_xyzs

        original_xyzs = torch.split(tensor=original_xyzs, split_size_or_sections=1, dim=1)
        original_xyzs = [torch.squeeze(input=original_xyz, dim=1).contiguous() for original_xyz in original_xyzs]

        if original_features is not None:
            original_features = torch.split(tensor=original_features, split_size_or_sections=1, dim=1)
            original_features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in original_features]

        new_features = []

        for t in range(T):
            dist, idx = three_nn(original_xyzs[t], xyzs[t])

            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feat = three_interpolate(features[t], idx, weight)

            if original_features is not None:
                new_feature = torch.cat([interpolated_feat, original_features[t]], dim=1)
            new_feature = self.conv(new_feature)
            new_features.append(new_feature)

        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features


@MODEL_REGISTRY.register()
class PSTTransformer(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        (radius, nsamples, spatial_stride, temporal_kernel_size,
         temporal_stride, dim, depths, heads, dim_head, mlp_dim, dropout1) = \
            (args.radius, args.nsamples, args.spatial_stride, args.temporal_kernel_size,
             args.temporal_stride, args.dim, args.depths, args.heads, args.dim_head, args.mlp_dim, args.dropout1)

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.use_transformer = True
        if self.use_transformer:
            self.transformer = Transformer(dim, depths, heads, dim_head, mlp_dim, dropout=dropout1)
        self.backbone_dim = dim

    def forward(self, input):  # [B, L, N, 3]
        
        xyzs, features = input[:, :, :, :3], None

        xyzs, features = self.tube_embedding(xyzs, features)  # [B, L, n, 3], [B, L, C, n]
        features = features.permute(0, 1, 3, 2)  # B, L, N, C

        if self.use_transformer:
            features = self.transformer(xyzs, features)

        out_xyz = xyzs.permute(0, 3, 1, 2).contiguous()
        out_feats = features.permute(0, 3, 1, 2).contiguous()  # B, C, L, N
        return out_xyz, out_feats, None


@MODEL_REGISTRY.register()
class PSTTransformerSemSeg(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        radius, nsamples = args.radius, args.nsamples

        self.conv1 = P4DConv(in_planes=3, mlp_planes=[32, 64, 128], mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                             spatial_kernel_size=[radius, nsamples], spatial_stride=4, 
                             temporal_kernel_size=1, temporal_stride=1, temporal_padding=[0,0],
                             operator='+', spatial_pooling='max', temporal_pooling='max')
        
        self.conv2 = P4DConv(in_planes=128, mlp_planes=[128, 128, 256], mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*radius, nsamples], spatial_stride=4, 
                             temporal_kernel_size=1, temporal_stride=1, temporal_padding=[0,0],
                             operator='+', spatial_pooling='max', temporal_pooling='max')
        
        self.conv3 = P4DConv(in_planes=256, mlp_planes=[256, 256, 512], mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*radius, nsamples], spatial_stride=4, 
                             temporal_kernel_size=3, temporal_stride=1, temporal_padding=[1,1],
                             operator='+', spatial_pooling='max', temporal_pooling='max')
        
        self.conv4 = P4DConv(in_planes=512, mlp_planes=[512, 512, 1024], mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*2*radius, nsamples], spatial_stride=2, 
                             temporal_kernel_size=1, temporal_stride=1, temporal_padding=[0,0],
                             operator='+', spatial_pooling='max', temporal_pooling='max')

        self.emb_relu = nn.ReLU()
        # self.transformer = Transformer(dim=1024, depths=2, heads=4, dim_head=256, mlp_dim=1024)
        self.backbone_dim = 128

        self.deconv4 = P4DTransConv(in_planes=1024, mlp_planes=[512, 512],
                                    mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True], 
                                    original_planes=512)

        self.deconv3 = P4DTransConv(in_planes=512, mlp_planes=[256, 256],
                                    mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                                    original_planes=256)

        self.deconv2 = P4DTransConv(in_planes=256, mlp_planes=[128, 128],
                                    mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                                    original_planes=128)

        self.deconv1 = P4DTransConv(in_planes=128, mlp_planes=[128, 128],
                                    mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                                    original_planes=3)
        
        self.outconv = nn.Conv2d(in_channels=128, out_channels=12, kernel_size=1, stride=1, padding=0)

    def forward(self, xyzs, rgbs):

        new_xyzs1, new_features1 = self.conv1(xyzs, rgbs)
        new_features1 = self.emb_relu(new_features1)

        new_xyzs2, new_features2 = self.conv2(new_xyzs1, new_features1)
        new_features2 = self.emb_relu(new_features2)

        new_xyzs3, new_features3 = self.conv3(new_xyzs2, new_features2)
        new_features3 = self.emb_relu(new_features3)

        new_xyzs4, new_features4 = self.conv4(new_xyzs3, new_features3)   # [B, L, n, 3], [B, L, C, n]
        new_features4 = self.emb_relu(new_features4).permute(0, 1, 3, 2).transpose(2, 3).contiguous()  # B, L, C, N

        # features = self.transformer(new_xyzs4, new_features4).transpose(2, 3).contiguous()  # B, L, C, N
        # new_features4 = self.emb_relu(features)

        new_xyzsd4, new_featuresd4 = self.deconv4(new_xyzs4, new_xyzs3, new_features4, new_features3)
        new_featuresd4 = self.emb_relu(new_featuresd4) + new_features3

        new_xyzsd3, new_featuresd3 = self.deconv3(new_xyzsd4, new_xyzs2, new_featuresd4, new_features2)
        new_featuresd3 = self.emb_relu(new_featuresd3) + new_features2

        new_xyzsd2, new_featuresd2 = self.deconv2(new_xyzsd3, new_xyzs1, new_featuresd3, new_features1)
        new_featuresd2 = self.emb_relu(new_featuresd2) + new_features1

        new_xyzsd1, new_featuresd1 = self.deconv1(new_xyzsd2, xyzs, new_featuresd2, rgbs)

        emb_features = new_featuresd1.transpose(1, 2)

        out = self.outconv(emb_features).transpose(1, 2)

        return out, emb_features.contiguous()
