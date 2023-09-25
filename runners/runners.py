import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from utils.utils import InputPadder
from utils.coords_and_warp import Coords
from utils.masks_and_occlusions import Occlusions
from losses import Loss


class BasicRunner(nn.Module):
    def __init__(self, model, args):
        super(BasicRunner, self).__init__()
        self.args = args

        self.model = model(args)
        self.get_coords = Coords()
        self.get_occ_mask = Occlusions(args.occlusions, args.use_full_size_warping)

        self.input_padder = InputPadder

        self.fwd = torch.tensor([0, 1], device=torch.device('cuda:'+str(args.gpu)))
        self.bwd = torch.tensor([1, 0], device=torch.device('cuda:'+str(args.gpu)))


    @torch.no_grad()
    def run_inference_step(self, example, fwd_bwd, suffix='_no_suffix_provided'):
        return self.model(example['ims'], return_last_flow_only=True, fwd_bwd=fwd_bwd, suffix=suffix)

    def forward(self, example, total_step=None, val_mode=False):

        if val_mode:
            outputs = self.run_inference_step(example, fwd_bwd=False, suffix='_pred')            

            return outputs, {}

        else:
            if self.training:

                outputs = self.run_training_step(example, total_step)
                loss_dict = self.loss(example, outputs, total_step)
                
                return outputs, loss_dict

            else:
                with torch.no_grad():

                    padder = self.input_padder(example['ims'].shape)
                    example['ims'] = padder.pad(example['ims'].flatten(end_dim=1)).unflatten(dim=0, sizes=(-1, 2))

                    outputs = self.run_training_step(example, total_step)

                    outputs['flow_f_pred'] = outputs['flows_aug'][:, 0, 0]
                
                    loss_dict = self.loss(example, outputs, total_step)
                
                return outputs, loss_dict


class Runner(BasicRunner):
    def __init__(self, model, args):
        super(Runner, self).__init__(model, args)

        self.loss = Loss(args)

    def run_training_step(self, example, total_step=None):

        pad_params = example['pad_params'].int() if self.args.use_full_size_warping else None

        # Forward pass of augmented images in the optical flow model
        outputs = self.model(example['ims_aug'], fwd_bwd=True, suffix='_aug')

        # Absolute coordinate computations
        outputs['coords'] = self.get_coords(outputs['flows_aug'])

        # Compute occlusion mask, boundary (out) mask with and without full-size wapring
        outputs['masks'], outputs['outs'], outputs['outs_full_size'] = \
            self.get_occ_mask(outputs['flows_aug'],            outputs['coords'], 
                              outputs['flows_aug'][:, self.bwd], outputs['coords'][:, self.bwd], 
                              example['masks_eraser'], pad_params=pad_params)

        if total_step >= self.args.selfsup_starting_step:
            # Forward pass of non-augmented images in the optical flow model to get teacher flow
            outputs.update(self.run_inference_step(example, fwd_bwd=True, suffix='_teacher'))
            # Forward pass of cropped augmented images in the optical flow model to get student flow
            outputs.update(self.model(example['ims_aug_stud'], fwd_bwd=True, suffix='_stud'))

        return outputs


