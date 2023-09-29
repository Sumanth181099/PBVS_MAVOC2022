import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import os
from torch.utils.data import ConcatDataset
import torchvision
from torchvision import transforms, models
import pdb
from numpy import nan
from pdb import set_trace as bp
from typing import Optional, Sequence
from torch import Tensor

device_ids=[0,1]
device = f'cuda:{device_ids[0]}'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def swd_single(p1,p2):
    p1 = p1.permute(1,2,0)
    p2 = p2.permute(1,2,0)
    h,w,c = p1.shape
    if c>1:
        proj = torch.randn(c,128).to(device)
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    sliced1,_ = torch.sort(p1)
    sliced2,_ = torch.sort(p2)
    dist = sliced1-sliced2
    wdist = torch.mul(dist, dist)
    #wdist = torch.mean(wdist)
    #wdist = torch.tensor([wdist], requires_grad=True)
    return wdist

def swd_diff(p1, p2):
    s = p1.shape
    p1 = p1.permute(0,2,3,1)
    p2 = p2.permute(0,2,3,1)
    if s[1]>1:
        #proj = torch.randn(s[1], 128).to(device)
        proj = torch.randn(s[1], 128).to('cuda:0')
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    sliced1,_ = torch.sort(p1)
    sliced2,_ = torch.sort(p2)
    dist = sliced1-sliced2
    wdist = torch.mul(dist, dist)
    wdist = torch.mean(wdist)
    return wdist

def swd(p1, p2):
    s = p1.shape
    #p1 = p1.permute(0,2,3,1)
    #p2 = p2.permute(0,2,3,1)
    if s[1]>1:
        proj = torch.randn(s[1], 128).to(device)
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1-p2
    wdist = torch.mean(torch.mul(dist, dist))
    return wdist

def ce_single(out,label):
    exp_class = torch.exp(out)
    if((exp_class==nan).any()):
        bp()
        exp_class[exp_class==nan] = 1e12
        bp()
    sum_exp = torch.sum(exp_class)
    o = exp_class/(sum_exp+1e-5)
    if((o==nan).any()):
        o[o==nan] = 1e12
        bp()
    if((o<=0.0).any()):
        o[o<=0.0] = 1e-12
    if((torch.log(o)!=nan).any()):
        try:  
            ce = -(label*(torch.log(o)))
        except: 
            bp()
    else:
        bp()
    return ce

class ce_loss(nn.Module):
    def forward(self,out,label):
        return ce_single(out,label)

class da_loss(nn.Module):
    def forward(self,feat1,feat2):
        return swd_diff(feat1,feat2)

class apple_da_loss(nn.Module):
    def forward(self,p1,p2):
        return swd(p1,p2)

class swd_loss(nn.Module):
    def forward(self,p1,p2):
        return swd_single(p1,p2)

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device=device,
               dtype=torch.float32) -> FocalLoss:
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl
