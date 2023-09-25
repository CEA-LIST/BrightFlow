import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mesh_grid(B, H, W, device):
    # mesh grid
    x_base = torch.arange(0, W, device=device).view(1, 1, -1).expand(B, H, -1)  # BHW
    y_base = torch.arange(0, H, device=device).view(1, -1, 1).expand(B, -1, W)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


class Coords(nn.Module):
    """ Get absolute coordinates from optical flow """
    def __init__(self) -> None:
        super(Coords, self).__init__()

    def get_coords(self, flow):
        _, _, H, W = flow.size()
        return mesh_grid(1, H, W, device=flow.device) + flow

    def forward(self, flows):
        dims = flows.size()
        return self.get_coords(flows.view(-1, 2, *dims[-2:])).view(dims)


class Warp(nn.Module):
    """ Warping module """
    def __init__(self) -> None:
        super(Warp, self).__init__()

    def warp(self, x, coords):
        B, _, H, W = coords.size()
        _coords = 2.0 * coords / torch.tensor([[[[max(W-1, 1)]], [[max(H-1, 1)]]]], device=coords.device).float() - 1.0
        _coords = _coords.permute(0, 2, 3, 1)
        
        x_warp = F.grid_sample(x, _coords, align_corners=True, padding_mode="zeros")

        mask = torch.ones(B, 1, H, W, requires_grad=False, device=_coords.device)
        mask = F.grid_sample(mask, _coords, align_corners=True, padding_mode="zeros")
        mask = (mask >= 1.0).float()

        return x_warp * mask

    def forward(self, x, coords):
        dims_x = x.size()
        x_warped = self.warp(x.view(-1, *dims_x[-3:]), coords.reshape(-1, 2, *dims_x[-2:]))
        return x_warped.view(*dims_x)


class WarpFullSize(Warp):
    """ Warping module implementing full-size warping """
    def __init__(self) -> None:
        super(WarpFullSize, self).__init__()
        self.coords = Coords()

    def forward(self, x, flows, pad_params, orig_dims):
        flow_dims = flows.size()
        H, W = flow_dims[-2:]
        x_warp = torch.empty((*flow_dims[:-3], x.size(-3), H, W), device=x.device)

        for b in range(flow_dims[0]):
            x_orig_dims = x[b, ..., :orig_dims[b, 0].data, :orig_dims[b, 1].data]
            coords_f_pad = self.coords(F.pad(flows[b], pad_params[b].tolist()))

            pad_left, _, pad_top, _ = pad_params[b].data
            x_warp[b] = self.warp(x_orig_dims, coords_f_pad)[..., pad_top:pad_top+H, pad_left:pad_left+W]
            
        return x_warp


class WarpMulti(Warp):
    """ Warp images multiple times with several flows """
    def __init__(self) -> None:
        super(WarpMulti, self).__init__()

    def forward(self, x, coords):
        coords_dims = coords.size() # B 2 S 2 H W or B 2 S N 2 H W
        H, W = coords_dims[-2:]
        C = x.size(-3)
        x_resized = x.unsqueeze(-4).expand(-1, -1, np.prod(coords_dims[2:-3]), -1, -1, -1).reshape(-1, C, H, W)
        coords_resized = coords.reshape(-1, *coords_dims[-3:])
        return self.warp(x_resized, coords_resized).view(*coords_dims[:-3], C, H, W)


class WarpFullSizeMulti(Warp):
    """ Warp images multiple times with several flows using full-size warping """
    def __init__(self) -> None:
        super(WarpFullSizeMulti, self).__init__()
        self.coords = Coords()

    def forward(self, x, flows, pad_params, orig_dims):
        flows_dims  = flows.size() # B 2 S 2 H W or B 2 S N 2 H W
        S = np.prod(flows_dims[2:-3]).astype(int) # S or S*N
        B = flows_dims[0]
        H, W = flows_dims[-2:]
        C = x.size(-3)
        x_warp = torch.empty((B, 2*S, C, H, W), device=x.device)

        for b in range(B):
            x_orig_dims = x[b, ..., :orig_dims[b, 0].data, :orig_dims[b, 1].data].unsqueeze(-4).expand(-1, S, -1, -1, -1).flatten(end_dim=-4)
            coords_f_pad = self.coords(F.pad(flows[b], pad_params[b].tolist()).flatten(end_dim=-4))

            pad_left, _, pad_top, _ = pad_params[b].data
            x_warp[b] = self.warp(x_orig_dims, coords_f_pad)[..., pad_top:pad_top+H, pad_left:pad_left+W]
            
        return x_warp.view(*flows_dims[:-3], C, H, W)
        