import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_corrections(img, correc, clamp_corrected_images=False):
    ''' Rescale corrections between 0 and 255 then apply corrections '''
    img_correc = img + correc*255
    if clamp_corrected_images:
        return torch.clamp(img_correc, 0, 255)
    else:
        return img_correc


def apply_corrections_uncropped(img_uncropped, correc, pad_params, clamp_corrected_images=False):
    ''' Rescale corrections between 0 and 255 then apply corrections on an uncropped image when full-size warping is used '''
    dims = correc.shape
    H, W = dims [-2:]
    H_uncropped, W_uncropped = img_uncropped.size()[-2:]
    img_correc = torch.empty_like(img_uncropped)
    for b in range(dims[0]):
        pad_left, _, pad_top, _ = pad_params[b].data
        correc_pad = F.pad(correc[b:b+1] * 255, (pad_left, W_uncropped - W - pad_left, pad_top, H_uncropped - H - pad_top))
        img_correc[b:b+1] = correc_pad + img_uncropped[b:b+1]

    if clamp_corrected_images:
        return torch.clamp(img_correc, 0, 255)
    else:
        return img_correc


@torch.no_grad()
def get_good_correction(im1, im2_warp, im2_warp_correc):
    '''Get a binary mask of the well-estimated corrections relative to L1-norm'''
    im1_diff = (im1 - im2_warp).abs().mean(dim=-3, keepdim=True)
    im1_diff_correc = (im1 - im2_warp_correc).abs().mean(dim=-3, keepdim=True)
    good_correc = im1_diff > im1_diff_correc
    return good_correc.float()


@torch.no_grad()
def get_best_index(coords, good_corrections):
    ''' Unwarp the good correction mask '''
    B, _, H, W = coords.size()
    coords_floor = torch.floor(coords)
    coords_offset = coords - coords_floor
    coords_floor = coords_floor.long()
    
    idx_batch_offset = torch.arange(B, device=coords.device).view(B, 1, 1).expand(-1, H, W) * H * W
        
    coords_floor_flattened = coords_floor.permute(0, 2, 3, 1).reshape(-1, 2)
    coords_offset_flattened = coords_offset.permute(0, 2, 3, 1).reshape(-1, 2)
    idx_batch_offset_flattened = idx_batch_offset.reshape(-1)
    good_corrections = good_corrections.reshape(-1)

    # Initialize results.
    idxs_list = []
    weights_list = []

    # Loop over differences di and dj to the four neighboring pixels.
    for di in range(2):
        for dj in range(2):
            idxs_i = coords_floor_flattened[:, 1] + di
            idxs_j = coords_floor_flattened[:, 0] + dj
            
            
            idxs = idx_batch_offset_flattened + idxs_i * W + idxs_j
            
            mask = torch.where(torch.logical_and(good_corrections.bool(), torch.logical_and(
                               torch.logical_and(idxs_i >= 0, idxs_i < H),
                               torch.logical_and(idxs_j >= 0, idxs_j < W))))[0]
                        
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
    
    counts = torch.zeros(B * H * W, device=coords.device).scatter_add(0, idxs, weights).reshape(B, 1, H, W)
            
    return counts > 0.