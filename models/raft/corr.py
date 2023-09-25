import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.utils import bilinear_sampler, coords_grid, positionalencoding2d

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, fwd_bwd=False):
        self.num_levels = num_levels
        self.radius = radius
        self.corrs_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        self.correlations = corr.view(batch, h1*w1, h2, w2)
        
        self.corrs_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corrs_pyramid.append(corr)


    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corrs_pyramid[i]

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        _, _, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht1*wd1)
        fmap2 = fmap2.view(batch, dim, ht2*wd2)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class CorrBlock_fb:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, fwd_bwd=False):
        self.num_levels = num_levels
        self.radius = radius
        self.corrs_pyramid = []

        # all pairs correlation
        corr1 = CorrBlock_fb.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr1.shape
        
        if fwd_bwd:
            corr2 = corr1.permute(0, 4, 5, 3, 1, 2)
            corrs = torch.stack([corr1, corr2], 1).reshape(2*batch*h1*w1, dim, h2, w2)
            self.correlations = torch.stack([corr1.view(batch, h1*w1, h2*w2), corr2.view(batch, h2*w2, h1*w1)], dim=1)

        else:
            corrs = corr1.reshape(batch*h1*w1, dim, h2, w2)
            self.correlations = corr1.view(batch, h1*w1, h2, w2)

        self.corrs_pyramid.append(corrs)
        for i in range(self.num_levels-1):
            corrs = F.avg_pool2d(corrs, 2, stride=2)
            self.corrs_pyramid.append(corrs)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corrs_pyramid[i]

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        _, _, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht1*wd1)
        fmap2 = fmap2.view(batch, dim, ht2*wd2)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class CorrBlock_fb_c2:
    def __init__(self, fmap1, fmap2, c2, num_levels=4, radius=4, fwd_bwd=False):
        self.num_levels = num_levels
        self.radius = radius
        self.corrs_pyramid = []
        self.c2_pyramid = []

        # all pairs correlation
        corr1 = CorrBlock_fb_c2.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr1.shape
        
        if fwd_bwd:
            corr2 = corr1.permute(0, 4, 5, 3, 1, 2)
            corrs = torch.stack([corr1, corr2], 1).reshape(2*batch*h1*w1, dim, h2, w2)
            self.correlations = torch.stack([corr1.view(batch, h1*w1, h2*w2), corr2.view(batch, h2*w2, h1*w1)], dim=1)

        else:
            corrs = corr1.reshape(batch*h1*w1, dim, h2, w2)
            self.correlations = corr1.view(batch, h1*w1, h2, w2)

        self.C = c2.size(1)
        c2 = c2.unsqueeze(1).expand(batch, h1*w1, self.C, h2, w2).reshape(batch*h1*w1, self.C, h2, w2)

        self.corrs_pyramid.append(corrs)
        for i in range(self.num_levels-1):
            corrs = F.avg_pool2d(corrs, 2, stride=2)
            self.corrs_pyramid.append(corrs)
            c2 = F.avg_pool2d(c2, 2, stride=2)
            self.c2_pyramid.append(c2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)

        out_pyramid = []
        out_c2_pyramid = []
        for i in range(self.num_levels):
            corr = self.corrs_pyramid[i]
            c2 = self.c2_pyramid[i]

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        _, _, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht1*wd1)
        fmap2 = fmap2.view(batch, dim, ht2*wd2)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr  / torch.sqrt(torch.tensor(dim).float())

class CorrC2Block:
    def __init__(self, fmap1, fmap2, c2, num_levels=4, radius=4, pe=False):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.c2_pyramid = []
        self.pe = positionalencoding2d(2*16, 2*self.radius+1, 2*self.radius+1).unsqueeze(0) if pe else torch.empty(1, 0, 2*self.radius+1, 2*self.radius+1)

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        _, self.C, _, _ = c2.size()
        c2 = c2.unsqueeze(1).expand(batch, h1*w1, self.C, h2, w2).reshape(batch*h1*w1, self.C, h2, w2)
        self.C = self.C + self.pe.size(1)
        
        self.corr_pyramid.append(corr)
        self.c2_pyramid.append(c2)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
            c2 = F.avg_pool2d(c2, 2, stride=2)
            self.c2_pyramid.append(c2)
            


    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        out_c2_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            c2 = self.c2_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr_softmax = F.softmax(corr.view(batch*h1*w1, 1, -1), -1).expand(batch*h1*w1, self.C, -1)
            c2 = torch.cat([bilinear_sampler(c2, coords_lvl), self.pe.expand(batch*h1*w1, -1, -1, -1)], 1)
            c2 = corr_softmax * c2.view(batch*h1*w1, self.C, -1)
            c2 = F.softmax(corr.view(batch*h1*w1, 1, -1), -1).expand(batch*h1*w1, self.C, -1) * c2.view(batch*h1*w1, self.C, -1)
            c2 = c2.sum(-1)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
            c2 = c2.view(batch, h1, w1, self.C)
            out_c2_pyramid.append(c2)

        out = torch.cat(out_pyramid, dim=-1)
        
        return out.permute(0, 3, 1, 2).contiguous().float(), out_c2_pyramid

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        _, _, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht1*wd1)
        fmap2 = fmap2.view(batch, dim, ht2*wd2) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
