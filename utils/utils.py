from turtle import forward
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.distributed as dist
from torch import Tensor


def to_cuda(example, excluded_keys=[]):
    for key, value in example.items():
        if key not in excluded_keys:
            if torch.is_tensor(value):
                example[key] = value.float().cuda()

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', div=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // div) + 1) * div - self.ht) % div
        pad_wd = (((self.wd // div) + 1) * div - self.wd) % div
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, inputs):
        return F.pad(inputs, self._pad, mode='replicate')

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def selfsup_transform_flow(x):
    '''Center-crop, resize and rescale flow for the selfsup loss'''
    _, _, H, W = x.size()
    x = x[..., 64:-64, 64:-64]
    _, _, H_down, W_down = x.size()
    x_up = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
    x_up[:, 0] *= W / W_down
    x_up[:, 1] *= H / H_down
    return x_up


@torch.no_grad()
def to_low_res(x, scale_factor=8):
    dims = x.size()
    H_new, W_new = dims[-2]//scale_factor, dims[-1]//scale_factor
    return torch.ceil(F.interpolate(x.view(-1, *dims[-3:]), (H_new, W_new), mode='area').view(*dims[:-2], H_new, W_new))


def l1(x):
    return torch.norm(x, p=1, dim=1, keepdim=True)


def rgb2gray(img):
    img_gray = img[..., 0, :, :]*0.2989 + img[..., 1, :, :]*0.1140 + img[..., 2, :, :]*0.5870
    return img_gray.unsqueeze(-3)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Logging:
    def __init__(self, runner, scheduler, args):
        self.args = args
        if args.gpu == 0:
            self.runner = runner
            self.scheduler = scheduler
            self.total_steps = args.init_step
            self.running_loss = {}
            self.LOG_FREQ = args.LOG_FREQ
            if not args.debug:
                log_dir = args.log_dir + datetime.now().strftime("%Y-%m-%d_%H-%M-%S" + '_' + args.name)
                print('log_dir:', log_dir)
                self.writer = SummaryWriter(log_dir=log_dir)
            else:
                warnings.warn("WARNING: debug mode activated no checkpoint will be saved")

            print("Parameter Count: %d" % count_parameters(runner))
            print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'start training')
    
    def _print_training_status(self):
        metrics_data = {k: round((self.running_loss[k].item() if type(self.running_loss[k]) == torch.Tensor else self.running_loss[k])/self.LOG_FREQ, 4) for k,v in self.running_loss.items()}
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        
        # print the training status
        print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), training_str, metrics_data)

        if not self.args.debug:
            for k in self.running_loss:
                self.writer.add_scalar(k, self.running_loss[k]/self.LOG_FREQ, self.total_steps)
                self.running_loss[k] = 0.0
        else:
            for k in self.running_loss:
                self.running_loss[k] = 0.0

    def push(self, metrics):
        if self.args.gpu == 0:
            self.total_steps += 1

            for key in metrics:
                if key not in self.running_loss:
                    self.running_loss[key] = 0.0

                self.running_loss[key] += metrics[key]

            if self.total_steps % self.LOG_FREQ == self.LOG_FREQ-1:
                self._print_training_status()
                self.running_loss = {}

    def write_dict(self, results):
        if self.args.gpu == 0:
            print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), results)
            if not self.args.debug:
                for key in results:
                    self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        if not self.args.debug and self.args.gpu == 0:
            self.writer.close()


@torch.no_grad()
def stack_all_gather_without_backprop(x: Tensor, dim: int = 0) -> Tensor:
    """Gather tensor across devices without grad.

    Args:
        x (Tensor): Tensor to gather.
        dim (int): Dimension to concat. Defaults to 0.

    Returns:
        Tensor: Gathered tensor.
    """
    if dist.is_available() and dist.is_initialized():
        tensors_gather = [torch.ones_like(x)
                          for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, x, async_op=False)
        output = torch.stack(tensors_gather, dim=dim)
    else:
        output = x
    return output

@torch.no_grad()
def list_all_gather_without_backprop(x: Tensor, dim: int = 0) -> Tensor:
    """Gather tensor across devices without grad.

    Args:
        x (Tensor): Tensor to gather.
        dim (int): Dimension to concat. Defaults to 0.

    Returns:
        Tensor: Gathered tensor.
    """
    if dist.is_available() and dist.is_initialized():
        tensors_gather = [torch.ones_like(x)
                          for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, x, async_op=False)
        output = tensors_gather
    else:
        output = [x]
    return output


class CheckpointSavior(object):
    def __init__(self, args):
        self.args = args
        self.main_metric = self.set_main_metric()
        self.best_metric = np.float32('inf')

    def set_main_metric(self):
        if 'kitti' in self.args.dataset_test.lower():
            return 'epe_occ'
        else:    
            return 'epe'

    def __call__(self, results, runner, custom_name=None):
        if self.args.gpu == 0:
            save_best_checkpoint = False

            if results[self.main_metric] < self.best_metric:
                self.best_metric = results[self.main_metric]
                save_best_checkpoint = True
                print(f'New best {self.main_metric}:', self.best_metric)

            else:
                print(f'Best {self.main_metric}:', self.best_metric)
                assert results[self.main_metric] <= 1e6, "Training interrupted because the model has diverged"

            if not self.args.debug:
                if save_best_checkpoint:
                    PATH = os.path.join(self.args.ckpt_dir, self.args.name, 'best_checkpoint.pth')
                    torch.save(runner.state_dict(), PATH)

                PATH = os.path.join(self.args.ckpt_dir, self.args.name, 'last_checkpoint.pth')
                torch.save(runner.state_dict(), PATH)

                if custom_name is not None:
                    PATH = os.path.join(self.args.ckpt_dir, self.args.name, custom_name + '.pth')
                    torch.save(runner.state_dict(), PATH)

            else:
                warnings.warn("WARNING: debug mode activated no checkpoint will be saved")
