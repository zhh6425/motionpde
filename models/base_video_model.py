import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch import Tensor
import numpy as np
import sys
import os
import math
from transformers import PreTrainedModel, PretrainedConfig
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils import *
from .build import *

TRAIN_MODE = {
    "encoder_only": 1,  # use encoder without solving
    "one_stage": 2,    # train encoder with solving end to end
    "two_stage": 3,    # pretrain encoder with solving
}

class Head(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(drop_rate)
    def forward(self, x):
        return self.linear(self.drop(x))
    
class SegHead(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(drop_rate)
    def forward(self, x):
        return self.linear(self.drop(x))

@MODEL_REGISTRY.register()
class BaseModel(PreTrainedModel):
    
    main_input_name = "inputs"
    config_class = PretrainedConfig

    def __init__(self, config, num_classes=None):
        super().__init__(config)
        self.encoder = build_model(config.model, config)

        config.backbone_dim = self.encoder.backbone_dim
        if 'Sequential' in self.encoder.__class__.__name__:
            in_dim = 4352
        else:
            in_dim = config.backbone_dim

        self.mode = TRAIN_MODE[config.train_mode]
        self.solving = build_model('MotionPDE', config) if self.mode > 1 else None
        self.loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
        self.head = Head(in_dim, num_classes, drop_rate=config.dropout_rate) if self.mode < 3 else None

    def forward(self, inputs, labels, indexs, dataset_name=None):  # [B, L, N, 3]
        out_dict = {'indexs': indexs, 'loss': 0}

        _, emb, end_points = self.encoder(inputs)  # batch, c, t, n
        out_dict['out_emb'] = emb

        if self.solving:
            out_dict = self.solving(out_dict)

        if 'PSTNet' in self.encoder.__class__.__name__:
            emb = torch.mean(input=out_dict['out_emb'], dim=-1, keepdim=False)
            emb = torch.max(input=emb, dim=-1, keepdim=False)[0]
        if 'Sequential' in self.encoder.__class__.__name__:
            emb = out_dict['out_emb'].contiguous().view(emb.size(0),-1)
            emb = torch.cat((emb, end_points[0], end_points[1]), 1)
        else:
            emb = torch.max(input=out_dict['out_emb'], dim=2, keepdim=False)[0]
            emb = torch.max(input=emb, dim=2, keepdim=False)[0]  

        if self.head:
            logit = self.head(emb)
            out_dict['logit'] = logit
            out_dict['loss'] += self.loss(logit, labels)

        return out_dict

@MODEL_REGISTRY.register()
class BaseActSegModel(PreTrainedModel):
    
    main_input_name = "inputs"
    config_class = PretrainedConfig

    def __init__(self, config, num_classes=None):
        super().__init__(config)
        self.encoder = build_model(config.model, config)

        config.backbone_dim = self.encoder.backbone_dim
        in_dim = config.backbone_dim
        self.num_classes = num_classes

        self.mode = TRAIN_MODE[config.train_mode]
        self.solving = build_model('MotionPDE', config) if self.mode > 1 else None
        self.loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
        self.head = Head(in_dim, num_classes, drop_rate=config.dropout_rate) if self.mode < 3 else None

    def forward(self, inputs, labels, indexs, dataset_name=None):  # [B, L, N, 3]
        out_dict = {'indexs': indexs, 'loss': 0}

        _, emb, end_points = self.encoder(inputs)  # batch, c, t, n
        out_dict['out_emb'] = emb

        if self.solving:
            out_dict = self.solving(out_dict)

        emb = torch.max(input=emb, dim=-1, keepdim=False)[0].transpose(1, 2)  # batch, t, c

        if self.head:
            logit = self.head(emb)
            out_dict['logit'] = logit
            if labels is not None:
                logit = logit.view(-1, self.num_classes)
                labels = labels.view(-1)
                out_dict['loss'] += self.loss(logit, labels)

        return out_dict

@MODEL_REGISTRY.register()
class BaseSemSegModel(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        self.use_info = args.use_info if args.use_info else False
        self.add_cross_proj = args.add_cross_proj if args.add_cross_proj else False
        self.encoder = build_model(args.model, args)

        args.backbone_dim = self.encoder.backbone_dim

        self.mode = TRAIN_MODE[args.train_mode]
        self.solving = build_model('MotionPDE', args) if self.mode > 1 else None

    def forward(self, input, dataset_name=None):  # [B, L, N, 3]
        out_dict = {}

        xyzs, rgbs = input

        logit, emb = self.encoder(xyzs, rgbs)  # batch, c, t, n
        out_dict['out_emb'] = emb

        if self.solving:
            out_dict = self.solving(out_dict)

        out_dict['logit'] = logit.transpose(1, 2)

        return out_dict


