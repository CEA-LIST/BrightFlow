import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock, CorrBlock_fb
from .utils.utils import bilinear_sampler, coords_grid, upflow8
from .gma import Attention, Aggregate

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTGMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.iters = args.iters

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='instance', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, ctxt):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        _, _, H, W = img.shape
        N = ctxt.size(0)
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def compute_flow(self, image, ctxt, corr_fn, flow_init=None, return_last_flow_only=False):

        net, inp = torch.split(ctxt, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        # attention, att_c, att_p = self.att(inp)
        attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image, ctxt)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            flow = coords1 - coords0

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(flow, mode=self.args.upsampling_mode)
            else:
                flow_up = self.upsample_flow(flow, up_mask)
            
            flow_predictions.append(flow_up)

        if return_last_flow_only:
            return flow_up
            
        flow_predictions.reverse()
        return flow_predictions
        

    def forward(self, images, flow_init=None, return_last_flow_only=False, fwd_bwd=True, suffix=''):
        """ Estimate optical flow between pair of frames """
        outputs = {}

        dims = images.size()
        images_norm = (2 * (images / 255.0) - 1.0).contiguous()
        images_norm_flatten = images_norm.flatten(end_dim=-4)

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmaps = self.fnet(images_norm_flatten).unflatten(dim=0, sizes=dims[:2])
        
        corrs_fn = CorrBlock_fb(fmaps[:, 0], fmaps[:, 1], radius=self.args.corr_radius, fwd_bwd=fwd_bwd)

        if fwd_bwd:
            with autocast(enabled=self.args.mixed_precision):
                ctxts = self.cnet(images_norm_flatten)

            flows = self.compute_flow(images_norm_flatten, ctxts, corrs_fn, return_last_flow_only=return_last_flow_only)

            if return_last_flow_only:
                outputs['flows' + suffix] = flows.unflatten(dim=0, sizes=dims[:2])
            else:
                outputs['flows' + suffix] = torch.stack(flows, dim=-4).unflatten(dim=0, sizes=dims[:2])

        else:
            images_norm1 = images_norm[:, 0]
            with autocast(enabled=self.args.mixed_precision):
                ctxt1 = self.cnet(images_norm1)

            outputs['flow_f' + suffix] = self.compute_flow(images_norm1, ctxt1, corrs_fn, flow_init=flow_init, return_last_flow_only=True)
            
        return outputs
