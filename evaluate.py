import os
import torch
import torch.nn as nn
import numpy as np

import utils.argument_parser as argument_parser
from utils.utils import InputPadder, to_cuda, list_all_gather_without_backprop
import utils.config as cfg


class Validation(object):
    def __init__(self, args):
        self.args = args
        self.dataset_test = args.dataset_test
        self.input_padder = InputPadder


    @torch.no_grad()
    def validate(self, runner, loader, metrics, total_step=0):

        dict_cumul = {}

        for _, example in enumerate(loader):

            ## Transfer to cuda
            to_cuda(example, excluded_keys=['orig_dims'])

            padder = self.input_padder(example['ims'].shape)
            example['ims'] = padder.pad(example['ims'].flatten(end_dim=1)).unflatten(dim=0, sizes=(-1, 2))

            output_dict, _ = runner(example=example, total_step=total_step, val_mode=True)

            metrics_dict = metrics(example, output_dict, padder)

            for key, value in metrics_dict.items():
                value_gathered = list_all_gather_without_backprop(value)
                if key in dict_cumul:
                    dict_cumul[key].extend(value_gathered)
                else:
                    dict_cumul[key] = value_gathered

        for key, values in dict_cumul.items():
            dict_cumul[key] = np.mean(torch.tensor(values).cpu().numpy())

        return dict_cumul


if __name__ == '__main__':

    ## get arguments
    args = argument_parser.get_arguments()
    args.gpu = 0

    assert args.batch_size == 1
    print(args)

    # set random seeds
    cfg.configure_random_seed(args.seed)

    ## get dataloaders
    test_loader = cfg.get_test_dataloaders(args)

    runner = cfg.get_runner(args, val_mode=True)
    print(sum(p.numel() for p in runner.parameters() if p.requires_grad))

    runner = nn.DataParallel(runner)
    runner.cuda()
    runner.eval()
    ckpt = torch.load(args.restore_ckpt)
    ckpt = {(k.replace('photometric_loss_function', 'census_loss') if 'photometric_loss_function' in k else k): v for k, v in ckpt.items()}
    missing_keys, unexpected_keys = runner.load_state_dict(ckpt, strict=False)
    print('missing_keys:', missing_keys)
    print('unexpected_keys:', unexpected_keys)

    metrics = cfg.get_metrics(args)

    validator = Validation(args)

    with torch.no_grad():
        res_dict = validator.validate(runner, test_loader, metrics)
        print(res_dict)