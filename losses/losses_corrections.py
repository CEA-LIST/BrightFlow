import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .losses import LossBasic
from utils.coords_and_warp import WarpMulti, WarpFullSizeMulti
from utils.utils_corrections import apply_corrections_uncropped, apply_corrections
from .loss_utils.photometric_loss_utils import PhotometricLossSequential, PhotometricLossCorrec


class LossCorrections(LossBasic):
    def __init__(self, args):
        super(LossCorrections, self).__init__(args)

        self.warp = WarpFullSizeMulti() if args.use_full_size_warping else WarpMulti()

        assert self.args.correc_starting_step <= self.args.correc_in_photo_starting_step, 'corrections should not be used if the corrector is not trained'

        self.photometric_loss = PhotometricLossSequential(args)
        self.photometric_loss_correc = PhotometricLossCorrec(args)

        self.apply_corrections = apply_corrections_uncropped if self.args.use_full_size_warping else apply_corrections

    def forward(self, example, outputs, total_step):

        self.update_selfsup_weight(total_step)

        loss_dict = {}

        if total_step >= self.args.correc_starting_step:

            masks = outputs['masks'][:, :, 0] * example['masks_eraser']

            # Apply correction on augmented images used in the correction loss
            # Then warp the corrected augmented images
            if self.args.use_full_size_warping:
                pad_params = example['pad_params'].int()
                orig_dims = example['orig_dims'].int()
                ims_aug_correc = self.apply_corrections(example['ims_aug_uncropped'], outputs['correcs_aug'] * masks, pad_params)
                ims_aug_warp_correc = self.warp(ims_aug_correc[:, self.bwd], outputs['flows_aug'][:, :, 0].detach(), pad_params, orig_dims)

            else:
                ims_aug_correc = self.apply_corrections(example['ims_aug'], outputs['correcs_aug'] * masks)
                ims_aug_warp_correc = self.warp(ims_aug_correc[:, self.bwd], outputs['coords'][:, :, 0].detach())

            # Apply correction losses
            loss_correc = self.photometric_loss_correc(example['ims_aug'], ims_aug_warp_correc, masks)
            
        else:
            loss_correc = torch.tensor(0.0, device=torch.device('cuda'))

        if total_step >= self.args.correc_in_photo_starting_step:
            # Apply correction on non-augmented images used in the photometric loss
            with torch.no_grad():
                masks = outputs['masks'][:, :, 0] * example['masks_eraser'] * outputs['best_indices']
                if self.args.use_full_size_warping:
                    pad_params = example['pad_params'].int()
                    orig_dims = example['orig_dims'].int()
                    example['ims_uncropped_correc'] = self.apply_corrections(example['ims_uncropped'], outputs['correcs'] * masks, pad_params, clamp_corrected_images=self.args.smart_clamp)

                else:
                    example['ims_correc'] = self.apply_corrections(example['ims'], outputs['correcs'] * masks, clamp_corrected_images=self.args.smart_clamp)

            # Reconstruction: warping of corrected images with flows predictions
            if self.args.use_full_size_warping:
                pad_params = example['pad_params'].int()
                orig_dims = example['orig_dims'].int()
                example['ims_warp'] = self.warp(example['ims_uncropped_correc'][:, self.bwd], outputs['flows_aug'], pad_params, orig_dims)
            else:
                example['ims_warp'] = self.warp(example['ims_correc'][:, self.bwd], outputs['coords'])

        else:

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
        loss_dict['smooth'] = loss_smooth
        loss_dict['self'] = loss_selfsup
        loss_dict['correc'] = loss_correc
        loss_dict['mean_mask'] = outputs['masks'][:, :, 0].mean()
        loss_dict['mean_flow'] = outputs['flows_aug'][:, 0, 0].abs().mean()
        if 'correcs_aug' in outputs:
            loss_dict['mean_correc'] = outputs['correcs_aug'][:, 0].mean()
        if 'best_indices' in outputs:
            loss_dict['best_indices'] = outputs['best_indices'][:, 0].float().mean()

        loss_dict['loss_total'] = loss_photo + self.selfsup_weight * loss_selfsup.clamp(0., 100.) + self.args.smoothness_weight * loss_smooth + self.args.correc_weight * loss_correc
        
        return loss_dict
