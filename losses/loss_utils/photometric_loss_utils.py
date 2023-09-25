import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from utils.utils import rgb2gray
from torch.cuda.amp import autocast
from utils.distances import l1, abs_robust_loss


class UnflowPhotometricLoss(object):
    def __init__(self, args):
        self.ssim_w = 0.85
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def SSIM(self, x, y, mu_x=None, mu_x_sq=None):

        if mu_x == None:
            mu_x = nn.AvgPool2d(3, 1)(x)
            mu_x_sq = torch.pow(mu_x, 2)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM = SSIM_n / SSIM_d

        SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

        return F.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)

    def __call__(self, image_1, image_2, im1_avg=None, im1_avg_sq=None):
        if self.ssim_w > 0 and self.ssim_w < 1:
            loss = (self.ssim_w * self.SSIM(image_1, image_2, im1_avg, im1_avg_sq) + (1-self.ssim_w) * l1(image_1 - image_2)).mean(dim=1, keepdim=True)
        elif self.ssim_w == 0:
            loss = l1(image_1 - image_2).mean(dim=1, keepdim=True)
        elif self.ssim_w == 1:
            loss = self.SSIM(image_1, image_2, im1_avg, im1_avg_sq).mean(dim=1, keepdim=True)
        else:
            raise ValueError('ssim_w should be in the intervalle (0,1)')
        return loss


class SSIM(object):
    def __init__(self):
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def __call__(self, x, y, mu_x=None, mu_x_sq=None):

        if mu_x == None:
            mu_x = nn.AvgPool2d(3, 1)(x)
            mu_x_sq = torch.pow(mu_x, 2)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM = SSIM_n / SSIM_d

        SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

        return F.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)


class CensusTransform(nn.Module):
    def __init__(self):
        super(CensusTransform, self).__init__()
        self.patch_size = 7
        self.num_pix_per_patch = self.patch_size ** 2
        self.pad_size = self.patch_size // 2
        self.conv2d = nn.Unfold(self.patch_size, padding=self.pad_size)
        # self.conv2d = nn.Conv2d(1, self.patch_size**2, self.patch_size, padding=self.pad_size, padding_mode='zeros', bias=False)
        # kernel = torch.eye(self.patch_size ** 2).view(self.patch_size, self.patch_size, 1, self.patch_size * self.patch_size).permute(3, 2, 1, 0)
        # self.conv2d.weight = nn.Parameter(kernel, requires_grad=False)

    def get_neighbors(self, x):
        dims = x.size()
        return self.conv2d(x.view(-1, *dims[-3:])).view(*dims[:-3], -1, *dims[-2:])
    
    def census(self, img):
        intensities = rgb2gray(img)
        neighbors = self.get_neighbors(intensities)
        diff = neighbors - intensities
        diff_norm = diff / (.81 + diff**2)**0.5
        return diff_norm

    def forward(self, x):
        return self.census(x)


class CensusLoss(CensusTransform):
    def __init__(self, args):
        super(CensusLoss, self).__init__()
        self.args = args
        self.sequence_weights = args.sequence_weight ** torch.arange(args.iters, dtype=torch.float, device=torch.device('cuda:'+str(args.gpu))).unsqueeze(0).unsqueeze(1)
        self.compute_loss = self.compute_loss_sequential if args.sequentially else self.compute_loss_parallel
        
    def soft_hamming(self, x, y, mask, thresh=.1):
        sq_dist = (x - y) ** 2
        soft_thresh_dist = sq_dist / (thresh + sq_dist)
        return soft_thresh_dist.sum(-3, keepdims=True), self.zero_mask_border(mask)

    def zero_mask_border(self, mask):
        """Used to ignore border effects from census_transform."""
        mask = mask[..., self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        return F.pad(mask, (self.pad_size, self.pad_size, self.pad_size, self.pad_size))

    def compute_loss_sequential(self, census_ims, ims_warp, masks):
        census_ims_warps = self.census(ims_warp)
        hamming, masks = self.soft_hamming(census_ims, census_ims_warps, masks)
        diff = (abs_robust_loss(hamming) * masks).sum((-3, -2, -1)) / (masks.sum((-3, -2, -1)) + 1e-5)
        return diff.mean()

    def compute_loss_parallel(self, ims, ims_warp, masks):
        census_ims = self.census(ims)
        census_ims_warps = self.census(ims_warp)
        hamming, masks = self.soft_hamming(census_ims.unsqueeze(-4), census_ims_warps, masks)
        diff = (abs_robust_loss(hamming) * masks).sum((-3, -2, -1)) / (masks.sum((-3, -2, -1)) + 1e-5)
        return (diff * self.sequence_weights).sum(-1).mean()

    def forward(self, ims, ims_warps, mask, covs=None):
        return self.compute_loss(ims, ims_warps, mask)


class PhotometricLoss(nn.Module):
    def __init__(self, args):
        super(PhotometricLoss, self).__init__()
        self.args = args
        if args.census_weight_flow > 0.:
            self.census_transform = CensusTransform()
            self.census_loss = CensusLoss(args)
        if self.args.ssim_weight_flow > 0.:
            self.ssim = SSIM()


class PhotometricLossCorrec(PhotometricLoss):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, ims, ims_warp, masks):
        loss = torch.tensor(0.0, device=torch.device('cuda'))

        # Apply requiered transformations on unwarped images
        if self.args.l1_weight_correc > 0. or self.args.ssim_weight_correc > 0.:
            ims_norm = ims / 255

        if self.args.census_weight_correc > 0.:
            census_ims = self.census_transform(ims)

        if self.args.ssim_weight_correc > 0.:
            ims_avg = nn.AvgPool2d(3, 1)(ims_norm.flatten(end_dim=-4))
            ims_avg_sq = torch.pow(ims_avg, 2)

        # Applying photometric losses
        if self.args.census_weight_correc > 0.:
            loss += self.args.census_weight_correc * self.census_loss(census_ims, ims_warp, masks)

        if self.args.ssim_weight_correc > 0.:
            loss += self.args.ssim_weight_correc * self.ssim(ims_norm, ims_warp/255, ims_avg, ims_avg_sq)

        if self.args.l1_weight_correc > 0.:
            loss += self.args.l1_weight_correc * ((torch.norm(ims_norm - ims_warp/255, p=1, dim=-3, keepdim=True) * masks).sum((-3, -2, -1)) / masks.sum((-3, -2, -1))).mean()
        
        return loss


class PhotometricLossSequential(PhotometricLoss):
    def __init__(self, args):
        super(PhotometricLossSequential, self).__init__(args)

    def forward(self, example, outputs):
        loss = torch.tensor(0.0, device=torch.device('cuda'))

        # Apply requiered transformations on unwarped images
        ims = example['ims']
        ims_norm = ims / 255

        if self.args.census_weight_flow > 0.:
            census_ims = self.census_transform(ims)

        if self.args.ssim_weight_flow > 0.:
            ims_avg = nn.AvgPool2d(3, 1)(ims_norm.flatten(end_dim=-4))
            ims_avg_sq = torch.pow(ims_avg, 2)

        for i in range(self.args.iters):
            current_seq_weight = self.args.sequence_weight ** i

            # Applying photometric losses
            if self.args.census_weight_flow > 0.:
                loss += self.args.census_weight_flow * current_seq_weight * self.census_loss(census_ims, example['ims_warp'][:, :, i], outputs['masks'][:, :, i])

            if self.args.ssim_weight_flow > 0.:
                loss += self.args.ssim_weight_flow * current_seq_weight * self.ssim(ims_norm, example['ims_warp'][:, :, i]/255, ims_avg, ims_avg_sq)

            if self.args.l1_weight_flow > 0.:
                loss += self.args.l1_weight_flow * current_seq_weight * ((torch.norm(ims_norm - example['ims_warp'][:, :, i]/255, p=1, dim=-3, keepdim=True) * outputs['masks'][:, :, i]).sum((-3, -2, -1)) / outputs['masks'].sum((-3, -2, -1))).mean()
        
        return loss


class PhotometricLossParallel(PhotometricLoss):
    def __init__(self, args):
        super(PhotometricLossParallel, self).__init__(args)
        self.args = args
        self.sequence_weights = args.sequence_weight ** torch.arange(args.iters, dtype=torch.float, device=torch.device('cuda:'+str(args.gpu))).unsqueeze(0).unsqueeze(1)

    def forward(self, example, outputs):
        loss = torch.tensor(0.0, device=torch.device('cuda'))

        # Applying photometric losses
        if self.args.census_weight_flow > 0.:
            loss += self.args.census_weight_flow * self.census_loss(example['ims'], example['ims_warp'], outputs['masks'])

        if self.args.ssim_weight_flow > 0.:
            raise NotImplementedError

        if self.args.l1_weight_flow > 0.:
            loss += self.args.l1_weight_flow * (self.sequence_weights * ((torch.norm(example['ims'] - example['ims_warp']/255, p=1, dim=-3, keepdim=True) * outputs['masks']).sum((-3, -2, -1)) / outputs['masks'].sum((-3, -2, -1)))).sum(-1).mean()

        return loss
    
