import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def robust_l1(x):
  """Robust L1 metric."""
  return (x**2 + 0.001**2)**0.5


def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return (torch.abs(diff) + eps) ** q


def l1(x):
    """L1 metric."""
    return torch.norm(x, p=1, dim=-3, keepdim=True)
