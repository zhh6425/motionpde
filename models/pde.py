import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import logging
import os
import sys
from timm.models.layers import trunc_normal_

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from .build import MODEL_REGISTRY


class SpectralCore(nn.Module):
    def __init__(self, width, head=8, num_basis=12, num_token=16, depth=1):
        super().__init__()
        self.width = width
        self.num_basis = num_basis
        self.head = head
        self.num_token = num_token
        self.depth = depth

        # basis
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)], dtype=torch.float)
        self.latent = nn.Parameter(torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        trunc_normal_(self.latent, std=.02)
        self.weights = nn.Parameter(torch.rand(self.width, self.num_basis * 2, dtype=torch.float))
        trunc_normal_(self.weights, std=.02)
        self.encode_linear = nn.Linear(self.width // self.head, self.width // self.head * 2)
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, emb, latent_token=None):
        B, N, C = emb.shape

        if latent_token is None:
            latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        emb = emb.view(B, -1, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()  # b, h, n, c/h
        k, v = self.encode_linear(emb).chunk(2, dim=-1)  # tuple of (b, h, n, c/h)
        attn = torch.einsum("bhlc,bhsc->bhls", latent_token, k)

        latent_token = torch.einsum("bhls,bhsc->bhlc", self.softmax(attn), v) + latent_token
        latent_token = latent_token.permute(0, 2, 1, 3).contiguous().view(B, -1, C)

        return latent_token

    def get_basis(self, x):
        # x: B, N, C
        self.modes_list = self.modes_list.to(x.device)
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, emb, weights):
        return torch.einsum("btcm,cm->btc", emb, weights)

    def transition(self, x):
        # transition
        attn_modes = self.get_basis(x)
        x = self.compl_mul2d(attn_modes, self.weights) + x
        return x

    def forward(self, emb):
        # b, n, c = emb.shape
        emb = self.attention(emb)
        # transition
        emb = self.transition(emb)
        return emb


@MODEL_REGISTRY.register()
class MotionPDE(nn.Module):
    def __init__(self, args, num_classes, **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}")

        self.c = args.hidden_dim
        self.projector = nn.Conv2d(args.backbone_dim, args.hidden_dim, 1, 1)
        # self.solve = nn.ModuleList([])
        self.solve = SpectralCore(args.hidden_dim, head=args.head, num_basis=args.num_basis, num_token=args.num_token, depth=args.depth)
        self.loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.loss_weight = args.loss_weight
        self.tem = args.tem

    def forward(self, in_dict):  # dict with input

        emb = self.projector(in_dict['out_emb']).permute(0, 2, 3, 1).contiguous()  # batch, t, n, c
        batch, t, n, c = emb.shape

        emb_t = emb.max(2)[0] # batch, t, c
        emb_n = emb.max(1)[0] # batch, n, c
        # emb_n = emb.view(batch, -1, c) # batch, tn, c

        emb_t = self.solve(emb_t)
        emb_n = self.solve(emb_n)
        # batch, num_token, c

        # contrastive on emb
        emb_t = F.normalize(emb_t, p=2, dim=-1).view(-1, self.c)
        emb_n = F.normalize(emb_n, p=2, dim=-1).view(-1, self.c)

        scores = torch.mm(emb_t, emb_n.transpose(0, 1)) / self.tem  # batch*num_token, batch*num_token
        labels = torch.tensor(range(scores.size(0)), dtype=torch.long, device=scores.device)
        loss = (self.loss(scores, labels) + self.loss(scores.transpose(0, 1), labels)) / 2

        in_dict['loss'] = in_dict['loss'] + loss * self.loss_weight
        # in_dict['scores'] = F.softmax(scores.view(batch, -1), dim=-1)
        
        return in_dict






# def casual_attention(self, emb, use_casual=False):
#         B, N, C = emb.shape

#         latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
#         emb = emb.view(B, -1, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()  # b, h, n, c/h
#         k, v = self.encode_linear(emb).chunk(2, dim=-1)  # tuple of (b, h, n, c/h)
#         attn = torch.einsum("bhlc,bhsc->bhls", latent_token, k)

#         # casual mask
#         if use_casual:
#             q_len, k_len = q.size(-2), k.size(-2)

#             causal_mask = torch.tril(
#                 torch.ones((q_len, k_len), dtype=torch.bool)
#             ).to(attn.device).view(1, 1, q_len, k_len)

#             mask_value = torch.finfo(attn.dtype).min
#             mask_value = torch.full([], mask_value, dtype=attn.dtype).to(attn.device)

#             attn = torch.where(causal_mask, attn, mask_value)

#         latent_token = torch.einsum("bhls,bhsc->bhlc", self.softmax(attn), v) + latent_token
#         latent_token = latent_token.permute(0, 2, 1, 3).contiguous().view(B, -1, C)

#         return latent_token

