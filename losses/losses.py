from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.coords_and_warp import WarpMulti, WarpFullSizeMulti
from .loss_utils.photometric_loss_utils import PhotometricLossSequential, PhotometricLossParallel
from .loss_utils.smoothness_loss_utils import SmoothnessLoss
from .loss_utils.selfsup_loss_utils import SelfSupLoss


class LossBasic(nn.Module):
    def __init__(self, args):
        super(LossBasic, self).__init__()
        self.args = args
        self.bwd = torch.tensor([1, 0], device=torch.device('cuda:'+str(args.gpu)))

        self.warp = WarpFullSizeMulti() if args.use_full_size_warping else WarpMulti()

        self.smoothness_loss = SmoothnessLoss(args)
        self.smoothness_weight = self.args.smoothness_weight

        self.selfsup_weight = 0.
        self.selfsup_loss = SelfSupLoss(args)

    def update_selfsup_weight(self, total_step):
        if total_step >= self.args.selfsup_starting_step:
            self.selfsup_weight = min(self.args.selfsup_weight_max, (total_step - self.args.selfsup_starting_step) / \
                                     (self.args.selfsup_end_rising_step - self.args.selfsup_starting_step) * self.args.selfsup_weight_max)


class Loss(LossBasic):
    def __init__(self, args):
        super(Loss, self).__init__(args)
        
        self.photometric_loss = PhotometricLossSequential(args) if args.sequentially else PhotometricLossParallel(args)

    def forward(self, example, outputs, total_step):

        self.update_selfsup_weight(total_step)

        loss_dict = {}

        # Reconstruction: warping of images with flows predictions
        if self.args.use_full_size_warping:
            pad_params = example['pad_params'].int()
            orig_dims = example['orig_dims'].int()
            example['ims_warp'] = self.warp(example['ims_uncropped'][:, self.bwd], outputs['flows_aug'], pad_params, orig_dims)
        else:
            example['ims_warp'] = self.warp(example['ims'][:, self.bwd], outputs['coords'])

        # Computation of the photometric loss
        loss_photo = self.photometric_loss(example, outputs)

        # Computation of the smoothness loss
        loss_smooth =  self.smoothness_loss(outputs['flows_aug'], example['ims'], example['masks_eraser'], total_step)

        # Computation of the selfsup loss
        loss_selfsup = torch.tensor(0.0, device=torch.device('cuda'))
        if total_step >= self.args.selfsup_starting_step:
            loss_selfsup += self.selfsup_loss(outputs['flows_stud'], outputs['flows_teacher'], example['masks_eraser_stud'], total_step)

        loss_dict['photo'] = loss_photo
        loss_dict['smoothness'] = loss_smooth
        loss_dict['selfsup'] = loss_selfsup
        loss_dict['mask_sum'] = outputs['masks'][:, :, 0].mean()
        loss_dict['mean_abs_flow'] = outputs['flows_aug'][:, 0, 0].abs().mean()

        loss_dict['loss_total'] = loss_photo + self.selfsup_weight * loss_selfsup.clamp(0., 100.) + self.args.smoothness_weight * loss_smooth
        
        return loss_dict
