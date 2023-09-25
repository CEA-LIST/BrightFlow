import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from enum import Enum

from utils.distances import robust_l1

def charbonnier(x, y, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return (robust_l1(x, y) + eps) ** q

def robust_l1(x, y):
  """Robust L1 metric."""
  return ((x - y)**2 + 0.001**2)**0.5

def huber_charbonnier(x, y, eps=1., q=0.4):
  """Home made mix between Charbonnier  and Huber loss"""
  return ((x - y)**2 + eps)**0.2


class Distances(Enum):
    L1 = 'l1'
    CHARBONNIER = 'charbonnier'
    HUBER = 'huber'
    CHARBONNIER_HUBER = 'huber_charbonnier'


class SelfSupLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sequence_weights = args.sequence_weight ** torch.arange(args.iters, dtype=torch.float, device=torch.device('cuda:'+str(args.gpu))).view(1, 1, args.iters)
        self.selfsup_loss = self.selfsup_loss_sequence_flows
        if args.selfsup_distance == 'l1':
            self.distance = robust_l1
        elif args.selfsup_distance == 'charbonnier':
            self.distance = charbonnier
        elif args.selfsup_distance == 'huber':
            self.distance = partial(F.huber_loss, reduction='none')
        elif args.selfsup_distance == 'huber_charbonnier':
            self.distance = huber_charbonnier

    def selfsup_transform_flow(self, x):
        dims = x.size()
        x = x.view(-1, *dims[-3:])
        H, W = dims[-2], dims[-1]
        x = x[..., 64:-64, 64:-64]
        _, _, H_down, W_down = x.size()
        x_up = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        x_up[:, 0] *= W / W_down
        x_up[:, 1] *= H / H_down
        return x_up.unflatten(dim=0, sizes=dims[:-3])
    
    def selfsup_loss_sequence_flows(self, flows_stud, flows_teacher, masks_eraser, weights=1.):
        flows_teacher = self.selfsup_transform_flow(flows_teacher)
        flows_teacher = flows_teacher.unsqueeze(2)
        masks_eraser = masks_eraser.unsqueeze(2)
        selfsup_loss = (self.distance(flows_teacher, flows_stud) * masks_eraser * weights).sum((-1, -2, -3)) / (masks_eraser.sum((-1, -2, -3)) + 1e-6)
        return ((selfsup_loss * self.sequence_weights).sum(-1)).mean()
    
    def selfsup_loss_last_flow(self, flows_stud, flows_teacher, masks_eraser, weights=1.):
        flows_teacher = self.selfsup_transform_flow(flows_teacher)
        selfsup_loss = (self.distance(flows_teacher, flows_stud[..., 0, :, :, :]) * masks_eraser).sum((-1, -2, -3)) / (masks_eraser.sum((-1, -2, -3)) + 1e-6)
        return selfsup_loss.mean()

    def forward(self, flows_stud, flows_teacher, masks_eraser, total_step, weights=1):
        return self.selfsup_loss_sequence_flows(flows_stud, flows_teacher, masks_eraser, weights)
