import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .coords_and_warp import Warp


class Occlusions(Warp):
    def __init__(self, occlusions, use_full_size_warping):
        super(Occlusions, self).__init__()

        if occlusions == 'brox':
            self.occlusions_estimator = self.occlusions_brox
        elif occlusions == 'wang':
            self.occlusions_estimator = self.occlusions_wang
        elif occlusions == 'none':
            self.occlusions_estimator = self.occlusions_none
        else:
            raise NotImplementedError

        if use_full_size_warping:
            self.get_occlusions_masks = self.occlusions_masks_full_size
        else:
            self.get_occlusions_masks = self.occlusions_masks

    def mask_out_flow(self, coords, margin=0):
        '''Mask boundary occlusions'''
        H, W = coords.size()[-2:]
        max_height, max_width = H-1-margin, W-1-margin
        mask = torch.logical_and(
            torch.logical_and(coords[..., 0:1, :, :] >= margin, coords[..., 0:1, :, :] <= max_width),
            torch.logical_and(coords[..., 1:2, :, :] >= margin, coords[..., 1:2, :, :] <= max_height))
        return mask


    def mask_out_flow_full_size_warp(self, coords, pad_params, margin=0):
        '''Mask boundary occlusions when full-size warping is available'''

        coords_dims = coords.size()
        pad_left, pad_right, pad_top, pad_bottom = torch.split(-pad_params.view(-1, 4, *(1,)*(len(coords_dims)-2)), 1, 1)
        pad_left += margin
        pad_right += margin
        pad_top += margin
        pad_bottom += margin
        H, W = coords_dims[-2:]
        max_height, max_width = H-1-pad_bottom, W-1-pad_right
        mask = torch.logical_and(
            torch.logical_and(coords[..., 0:1, :, :] >= pad_left, coords[..., 0:1, :, :] <= max_width),
            torch.logical_and(coords[..., 1:2, :, :] >= pad_top, coords[..., 1:2, :, :] <= max_height))
        return mask

    def occlusions_none(self, *args, **kwargs):
        return 1.

    def occlusions_brox(self, forward_flow, forward_coords, backward_flow, backward_coords):
        # Warp backward flow
        dims = backward_flow.size()
        backward_flow_warp = self.warp(backward_flow.reshape(-1, *dims[-3:]), forward_coords.reshape(-1, *dims[-3:])).view(*dims)

        # Compute occlusions based on forward-backward consistency.
        fb_sq_diff = torch.sum((forward_flow + backward_flow_warp)**2, axis=-3, keepdims=True)
        fb_sum_sq = torch.sum(forward_flow**2 + backward_flow_warp**2, axis=-3, keepdims=True)

        occ = fb_sq_diff < 0.01 * fb_sum_sq + 0.5

        return occ.view(*dims[:-3], 1, *dims[-2:])

    def occlusions_wang(self, forward_flow, forward_coords, backward_flow, backward_coords):
        dims = backward_coords.size()
        B, H, W = np.prod(dims[:-3]), dims[-2], dims[-1]
        backward_coords = backward_coords.reshape(-1, *dims[-3:])
        coords_floor = torch.floor(backward_coords)
        coords_offset = backward_coords - coords_floor
        coords_floor = coords_floor.long()
        
        idx_batch_offset = torch.arange(B, device=backward_coords.device).view(B, 1, 1).expand(-1, H, W) * H * W
            
        coords_floor_flattened = coords_floor.permute(0, 2, 3, 1).reshape(-1, 2)
        coords_offset_flattened = coords_offset.permute(0, 2, 3, 1).reshape(-1, 2)
        idx_batch_offset_flattened = idx_batch_offset.reshape(-1)

        # Initialize results.
        idxs_list = []
        weights_list = []

        # Loop over differences di and dj to the four neighboring pixels.
        for di in range(2):
            for dj in range(2):
                idxs_i = coords_floor_flattened[:, 1] + di
                idxs_j = coords_floor_flattened[:, 0] + dj
                
                
                idxs = idx_batch_offset_flattened + idxs_i * W + idxs_j
                
                mask = torch.where(torch.logical_and(torch.logical_and(idxs_i >= 0, idxs_i < H),
                                                                torch.logical_and(idxs_j >= 0, idxs_j < W)))[0]
                
                valid_idxs = torch.gather(idxs, 0, mask)
                valid_offsets = torch.stack([torch.gather(coords_offset_flattened[:, 0], 0, mask), torch.gather(coords_offset_flattened[:, 1], 0, mask)], 1)
                            
                # Compute weights according to bilinear interpolation.
                weights_i = (1. - di) - (-1)**di * valid_offsets[:, 1]
                weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 0]
                weights = weights_i * weights_j

                # Append indices and weights to the corresponding list.
                idxs_list.append(valid_idxs)
                weights_list.append(weights)
                
        # Concatenate everything.
        idxs = torch.cat(idxs_list, 0)
        weights = torch.cat(weights_list, 0)
        
        counts = torch.zeros(B * H * W, device=backward_coords.device).scatter_add(0, idxs, weights).reshape(B, 1, H, W).clamp(0, 1)
                
        return counts.view(*dims[:-3], 1, H, W)

    def occlusions_masks(self, flow_f, coords_f, flow_b, coords_b, masks_eraser=None, pad_params=None):
        '''Get final occlusion masks'''
        mask_out = self.mask_out_flow(coords_f)
        occ = self.occlusions_estimator(forward_flow=flow_f, forward_coords=coords_f, backward_flow=flow_b, backward_coords=coords_b)
        if len(masks_eraser.size()) + 1 == len(mask_out.size()):
            masks_eraser = masks_eraser.unsqueeze(-4)
        return mask_out * occ * masks_eraser, mask_out, None

    def occlusions_masks_full_size(self, flow_f, coords_f, flow_b, coords_b, masks_eraser=None, pad_params=None):
        '''Get final occlusion masks when full-size warping is available'''
        mask_out = self.mask_out_flow(coords_f)
        mask_out_pad = self.mask_out_flow_full_size_warp(coords_f, pad_params)
        occ = self.occlusions_estimator(forward_flow=flow_f, forward_coords=coords_f, backward_flow=flow_b, backward_coords=coords_b) + ~mask_out
        if len(masks_eraser.size()) + 1 == len(mask_out.size()):
            masks_eraser = masks_eraser.unsqueeze(-4)
        return mask_out_pad * occ.float().clamp(0,1) * masks_eraser, mask_out, mask_out_pad

    @torch.no_grad()
    def forward(self, flow_f, coords_f, flow_b, coords_b, masks_eraser=None, pad_params=None):
        return self.get_occlusions_masks(flow_f, coords_f, flow_b, coords_b, masks_eraser, pad_params)
