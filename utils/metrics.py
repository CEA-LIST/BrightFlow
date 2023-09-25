import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Metrics(object):
    def __init__(self, args, dataset_name):
        self.args = args
        if 'kitti' in dataset_name.lower():
            self.data_metric = self.kitti_metrics
        elif 'hd1k' in dataset_name.lower():
            self.data_metric = self.kitti_metrics
        else:
            self.data_metric = self.standard_metrics

        self.flow_metrics = self.epe_f1


    def epe_f1(self, flow_pred, flow_gt, mask, suffix=''):
        ''' Compute EPE and F1 (or %ER) metrics'''

        epe = torch.sum((flow_pred - flow_gt)**2, dim=1, keepdims=True).sqrt()
        f1 = torch.logical_and(epe > 3, epe / torch.sum(flow_gt**2, dim=1, keepdims=True).sqrt() > 0.05).float()
        epe = (epe * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
        f1 = (f1 * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)) * 100

        return {
            'epe'+suffix: epe.mean(), 
            'f1'+suffix: f1.mean()
        }
            
    def kitti_metrics(self, example, flow_pred):

        flow_occ = example['flow_occ']
        valid_occ = example['valid_occ']

        flow_noc = example['flow_noc']
        valid_noc = example['valid_noc']

        metrics_dict = {}
        metrics_dict.update(self.flow_metrics(flow_pred, flow_occ, valid_occ, suffix='_occ'))
        metrics_dict.update(self.flow_metrics(flow_pred, flow_noc, valid_noc, suffix='_noc'))

        return metrics_dict

    def standard_metrics(self, example, flow_pred):

        flow_gt = example['flow']
        valid = example['valid']

        metrics_dict = self.flow_metrics(flow_pred, flow_gt, valid)

        return metrics_dict

    def resize_flow(self, flow, new_size):
        H, W = flow.size()[2:]
        new_H, new_W = new_size
        flow = F.interpolate(flow, new_size, mode='bilinear', align_corners=True)
        return flow * torch.tensor([new_W/W, new_H/H], device=flow.device).reshape(1, 2, 1, 1)

    def __call__(self, example, output_dict, padder):
        
        metrics = {}
        flow_pred = padder.unpad(output_dict['flow_f_pred'])
        metrics.update(self.data_metric(example, flow_pred))

        return metrics

