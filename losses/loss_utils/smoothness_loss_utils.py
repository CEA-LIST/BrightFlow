import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from utils.distances import robust_l1


class SmoothnessLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sequence_weights = args.sequence_weight ** torch.arange(args.iters, dtype=torch.float, device=torch.device('cuda:'+str(args.gpu))).view(1, 1, args.iters)
        self.smooth_const = -150
        if args.smoothness_order == 1:
            self.flow_grad = self.flow_grad_1st_order
        elif args.smoothness_order == 2:
            self.flow_grad = self.flow_grad_2nd_order

    def grad(self, img, stride=1):
        gx = img[..., :, :-stride] - img[..., :, stride:]  # NCHW
        gy = img[..., :-stride, :] - img[..., stride:, :]  # NCHW
        return gx, gy

    def grad_img(self, im, stride):
        im_grad_x, im_grad_y = self.grad(im, stride)
        im_grad_x = im_grad_x.abs().mean(-3, keepdim=True)
        im_grad_y = im_grad_y.abs().mean(-3, keepdim=True)
        return  im_grad_x, im_grad_y

    def get_smoothness_mask(self, mask, stride=1):
        mask_x = mask[..., :-stride] * mask[..., stride:]
        mask_y = mask[..., :-stride, :] * mask[..., stride:, :]
        return mask_x, mask_y
    
    def flow_grad_1st_order(self, flows):
        return self.grad(flows)
    
    def flow_grad_2nd_order(self, flows):
        flows_grad_x, flows_grad_y = self.grad(flows)
        flows_grad_xx, _ = self.grad(flows_grad_x)
        _, flows_grad_yy = self.grad(flows_grad_y)
        return flows_grad_xx, flows_grad_yy
    
    def smoothness_loss_last_flow(self, flows, ims, masks):

        ims_grad_x, ims_grad_y = self.grad_img(ims, self.args.smoothness_order)
        mask_x, mask_y = self.get_smoothness_mask(masks, stride=self.args.smoothness_order)

        flows_grad_x, flows_grad_y = self.flow_grad(flows[..., 0, :, :, :])
        smoothness_loss = ((torch.exp(self.smooth_const * ims_grad_x.abs().mean(-3, keepdim=True)) * robust_l1(flows_grad_x) * mask_x).sum((-1, -2, -3)) / (mask_x.sum((-1, -2, -3)) + 1e-6) +
                           (torch.exp(self.smooth_const * ims_grad_y.abs().mean(-3, keepdim=True)) * robust_l1(flows_grad_y) * mask_y).sum((-1, -2, -3)) / (mask_y.sum((-1, -2, -3)) + 1e-6)) / 2
        return smoothness_loss.mean()
    
    def smoothness_loss_sequence_flow(self, flows, ims, masks):
        ims_grad_x, ims_grad_y = self.grad_img(ims.unsqueeze(2), self.args.smoothness_order)
        
        mask_x, mask_y = self.get_smoothness_mask(masks.unsqueeze(2), stride=self.args.smoothness_order)

        flows_grad_x, flows_grad_y = self.flow_grad(flows)
        smoothness_loss = ((torch.exp(self.smooth_const * ims_grad_x.abs().mean(-3, keepdim=True)) * robust_l1(flows_grad_x) * mask_x).sum((-1, -2, -3)) / (mask_x.sum((-1, -2, -3)) + 1e-6) +
                           (torch.exp(self.smooth_const * ims_grad_y.abs().mean(-3, keepdim=True)) * robust_l1(flows_grad_y) * mask_y).sum((-1, -2, -3)) / (mask_y.sum((-1, -2, -3)) + 1e-6)) / 2
        return ((smoothness_loss * self.sequence_weights).sum(2)).mean()
    
    def forward(self, flows, ims, masks, total_step):
        return self.smoothness_loss_sequence_flow(flows, ims, masks)
