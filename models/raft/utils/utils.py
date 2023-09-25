import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width).cuda()
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(np.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, height, -1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, height, -1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, width)
    idx = torch.cat([torch.LongTensor(range(d_model//4)), torch.LongTensor(range(d_model//2,d_model//2 + d_model//4))])

    return pe[idx]


def mask_corr(pad_size, dims):
    H, W = dims
    top = torch.zeros(1, pad_size, H, W, device=torch.device('cuda'))
    bottom = torch.zeros(1, pad_size, H, W, device=torch.device('cuda'))
    left = torch.zeros(1, pad_size, H, W, device=torch.device('cuda'))
    right = torch.zeros(1, pad_size, H, W, device=torch.device('cuda'))

    for w in range(W):
        for i in range(pad_size):
            if w <= i:
                left[:, i, :, w] = 1
            if W-w <= i+1:
                right[:, i, :, w] = 1

    for h in range(H):
        for i in range(pad_size):
            if h <= i:
                top[:, i, h] = 1
            if H-h <= i+1:
                bottom[:, i, h] = 1

    return torch.cat([top, bottom, left, right], dim=1)


def mask_padding(pad_size, like):
    zeros = torch.zeros_like(like)
    return F.pad(zeros, (pad_size, pad_size, pad_size, pad_size), value=1.)


class PadFmaps(nn.Module):
    def __init__(self, pad_size, ch, dims):
        super(PadFmaps, self).__init__()
        self.pad_size = pad_size
        self.smart_pad = nn.Sequential(*[nn.ConvTranspose2d(ch, ch, 3, stride=1), nn.LeakyReLU(0.1, inplace=True), 
                                         nn.ConvTranspose2d(ch, ch, 3, stride=1), nn.LeakyReLU(0.1, inplace=True)]*(pad_size//2))
        self.zero_pad = nn.ZeroPad2d(pad_size)

    def forward(self, fmap):
        zeros_pad = self.zero_pad(fmap)
        smart_pad = self.smart_pad(fmap)
        return zeros_pad + mask_padding(self.pad_size, fmap).to(smart_pad.device) * smart_pad


def get_out_corrs(coords, pad_size, radius):
    B, _, H, W = coords.size()
    coords = coords - pad_size
    coords_h = coords[:, 1]
    coords_w = coords[:, 0]
    coords_h[coords_h==torch.clamp(coords_h, radius , H - radius)]=0
    coords_w[coords_w==torch.clamp(coords_w, radius , W - radius)]=0
    coords = coords - coords_grid(B, H, W, coords.device) * (coords != 0)
    return F.hardtanh(coords/pad_size)
    