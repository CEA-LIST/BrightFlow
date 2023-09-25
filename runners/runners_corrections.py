import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast

from .runners import BasicRunner
from models.raft.corrector import Corrector
from losses import LossCorrections
from utils.utils_corrections import apply_corrections_uncropped, apply_corrections, get_best_index, get_good_correction
from utils.coords_and_warp import Warp, WarpFullSize


class RunnerCorrection(BasicRunner):
    def __init__(self, model, args):
        super(RunnerCorrection, self).__init__(model, args)

        self.warp = WarpFullSize() if args.use_full_size_warping else Warp()
        
        self.corrector = Corrector(args, intput_dim=args.input_dim_corrector, norm_fn='instance')
        self.loss = LossCorrections(args)
        self.apply_corrections = apply_corrections_uncropped if self.args.use_full_size_warping else apply_corrections


    @torch.no_grad()
    def get_corrector_inputs(self, ims, ims_warp, masks=None, outs=None, outs_full_size=None):
        '''Set the inputs for the correction estimator'''
        inputs_correc = [2*(ims/255.) - 1., 2*(ims_warp/255.) - 1.]

        if self.args.occ_in_correc_inputs:
            inputs_correc.append(masks)
        elif self.args.no_FSW_occ_in_correc_inputs:
            inputs_correc.append(masks * outs)
        elif self.args.occ_and_out_in_correc_inputs:
            inputs_correc.extend([masks, outs])
        
        return torch.cat(inputs_correc, dim=-3)

    def run_training_step(self, example, total_step=None):
        outputs = {}

        B, _, _, H, W = example['ims'].size()

        if self.args.use_full_size_warping:
            pad_params = example['pad_params'].int()
            orig_dims = example['orig_dims'].int()
        else:
            pad_params = None

        # Forward pass of augmented images in the optical flow model
        outputs = self.model(example['ims_aug'], fwd_bwd=True, suffix='_aug')

        # Absolute coordinate computations
        outputs['coords'] = self.get_coords(outputs['flows_aug'])

        # Compute occlusion mask, boundary (out) mask with and without full-size wapring
        outputs['masks'], outputs['outs'], outs_full_size = \
            self.get_occ_mask(outputs['flows_aug'][:, self.fwd], outputs['coords'][:, self.fwd], 
                              outputs['flows_aug'][:, self.bwd], outputs['coords'][:, self.bwd], 
                              example['masks_eraser'], pad_params=pad_params)

        if total_step >= self.args.selfsup_starting_step:
            # Forward pass of non-augmented images in the optical flow model to get teacher flow
            outputs.update(self.run_inference_step(example, fwd_bwd=True, suffix='_teacher'))
            # Forward pass of cropped augmented images in the optical flow model to get student flow
            outputs.update(self.model(example['ims_aug_stud'], fwd_bwd=True, suffix='_stud'))

        if total_step >= self.args.correc_starting_step:

            with torch.no_grad():

                # Warp augmented image
                if self.args.use_full_size_warping:
                    ims_aug_warp_0 = self.warp(example['ims_aug_uncropped'][:, self.bwd], outputs['flows_aug'][:, :, 0], pad_params, orig_dims)

                else:
                    ims_aug_warp_0 = self.warp(example['ims_aug'][:, self.bwd], outputs['coords'][:, :, 0])

                # Set the augmented inputs for the corrector
                inputs_correc_aug = self.get_corrector_inputs(ims=example['ims_aug'], ims_warp=ims_aug_warp_0,
                                                                                 masks=outputs['masks'][:, :, 0], outs=outputs['outs'][:, :, 0], outs_full_size=outs_full_size[:, :, 0])

            with autocast(enabled=self.args.mixed_precision):
                # Forward pass in the corrector with augmented inputs
                outputs['correcs_aug'] = self.corrector(inputs_correc_aug.detach())

            if total_step >= self.args.correc_in_photo_starting_step:

                with torch.no_grad():
                    flows0 = outputs['flows_aug'][:, :, 0]
                    coords0 = outputs['coords'][:, :, 0]
                    masks0 = outputs['masks'][:, :, 0]
                    outs0 = outputs['outs'][:, :, 0]
                    outs_full_size0 = outs_full_size[:, :, 0]

                    # Warp non-augmented image
                    if self.args.use_full_size_warping:
                        ims_warp_0 = self.warp(example['ims_uncropped'][:, self.bwd], flows0, pad_params, orig_dims)

                    else:
                        ims_warp_0 = self.warp(example['ims'][:, self.bwd], coords0)

                    # Set the non-augmented inputs for the corrector
                    inputs_correc = self.get_corrector_inputs(ims=example['ims'], ims_warp=ims_warp_0,
                                                              masks=masks0, outs=outs0, outs_full_size=outs_full_size0)
                    
                    with autocast(enabled=self.args.mixed_precision):
                    # Forward pass in the corrector with non-augmented inputs
                        correcs = self.corrector(inputs_correc.detach())
                
                outputs['correcs'] = correcs

                if self.args.keep_good_corrections_only:

                    with torch.no_grad():

                        # Warp non-augmented image
                        if self.args.use_full_size_warping:
                            pad_params = example['pad_params'].int()
                            orig_dims = example['orig_dims'].int()
                            ims_correc = self.apply_corrections(example['ims_uncropped'], outputs['correcs'], pad_params, clamp_corrected_images=self.args.smart_clamp)
                            ims_warp_correc_0 = self.warp(ims_correc[:, self.bwd], outputs['flows_aug'][:, :, 0].detach(), pad_params, orig_dims)

                        else:
                            ims_correc = self.apply_corrections(example['ims'], outputs['correcs'], clamp_corrected_images=self.args.smart_clamp)
                            ims_warp_correc_0 = self.warp(ims_correc[:, self.bwd], outputs['coords'][:, :, 0].detach())

                        # Get well-estimated correction mask
                        good_correcs = get_good_correction(example['ims'], ims_warp_0, ims_warp_correc_0)

                        # Unwarp the well-estimated correction mask 
                        outputs['best_indices'] = get_best_index(coords0.flatten(end_dim=-4), good_correcs.flatten(end_dim=-4)).unflatten(dim=0, sizes=(B, 2))[:, self.bwd]

                else:
                    outputs['best_indices'] = torch.ones_like(masks0)

        return outputs
