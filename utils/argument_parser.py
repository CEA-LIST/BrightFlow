import argparse
import json
import os
import torch

def get_arguments():
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--name', type=str, default='raft', help="name your experiment")
    add('--ckpt_dir', type=str)
    add('--log_dir', type=str)
    add('--restore_ckpt', type=str, default=None)
    add('--init_step', type=int, default=0)
    add('--dataset_train', type=str) #, choices=['Chairs', 'Sintel', 'KITTI', 'HD1K']
    add('--dataset_test', type=str)  #, choices=['Chairs', 'Sintel', 'KITTI', 'HD1K']
    add('--data_in_grayscale', action='store_true', help='indicate inputs are in grayscale')
    add('--num_workers', type=int, default=4)
    add('--seed', type=int, default=1)
    add('--debug', action='store_true', help='if True, no checkpoint file will be created')
    add('--sequentially', action='store_true', help='Apply photometric loss function in a sequential way to reduce memory cost')
    add('--VAL_FREQ', type=int, default=1000, help='validation frequency')
    add('--LOG_FREQ', type=int, default=100, help='log frequency')

    add('--batch_size', type=int)
    add('--num_steps', type=int)

    #Evaluation
    add('--eval', type=str, nargs='+', choices=['flow', 'match'], default='flow')
    add('--eval_on_train', action='store_true', help='Evaluate on train set')

    #Optimization
    add('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam')
    add('--lr', type=float, default=0.0002)
    add('--lr_decay_max', type=float, default=0.001)
    add('--scheduler', type=str, choices=['smurf', 'raft'], default='smurf')
    add('--end_warmup_step', type=int, default=0)
    add('--lr_decay_step', type=int)
    add('--wdecay', type=float, default=.00005)
    add('--epsilon', type=float, default=1e-8)
    add('--clip', type=float, default=1.0)
    add('--mixed_precision', action='store_true', help='use mixed precision')
    add('--sequence_weight', type=float, default=0.8)

    #Model
    add('--mode', type=str, choices=['flow_correc', 'flow_only'])
    add('--model', type=str, choices=['raft', 'gma', 'scv'])
    add('--small', action='store_true', help='use small model if model is raft')
    add('--iters', type=int, default=12)
    add('--dim_out_flow', type=int, default=2)
    add('--dropout', type=float, default=0.0)
    add('--crop_size', type=int, nargs='+', default=None)
    add('--occlusions', type=str, choices=['wang', 'brox', 'none'], default=None)
    add('--pad_mode', type=str, choices=['zeros', 'replicate', 'reflect'], default='zeros')
    add('--upsampling_mode', type=str, choices=['bilinear', 'bicubic', 'convex'], help='flow upsampling', default='convex')

    #Augmentations
    add('--no_photo_aug', action='store_true', help='To train the network without data augmentation (only works with corrections')
    add('--random_eraser', action='store_true', help='Use random eraser eraser')

    #Photometric loss
    add('--census_weight_flow', type=float, default=0.0)
    add('--census_weight_correc', type=float, default=0.0)
    add('--unflow_weight_flow', type=float, default=0.0)
    add('--unflow_weight_correc', type=float, default=0.0)
    add('--l1_weight_flow', type=float, default=0.0)
    add('--l1_weight_correc', type=float, default=0.0)
    add('--census_patch_size', type=int, default=7)
    add('--use_full_size_warping', action='store_true')

    #Smoothness loss
    add('--smoothness_order', type=int, choices=[1, 2], help='Order of the gradient of the flow used in the smoothness loss')
    add('--smoothness_weight', type=float)

    #Selfsup loss
    add('--selfsup_starting_step', type=int)
    add('--selfsup_end_rising_step', type=int)
    add('--selfsup_weight_max', type=float)
    add('--selfsup_distance', type=str, choices=['l1', 'charbonnier', 'huber', 'huber_charbonnier'], default='l1', help='Distance used to compare the flow from the teacher and the flow from the student')

    #Correction
    add('--correc_in_photo_starting_step', type=int, default=0)
    add('--correc_starting_step', type=int, default=0)
    add('--correc_weight', type=float, default=0.0)
    add('--smart_clamp', action='store_true', help='clip corrected images in the photometric loss')
    add('--keep_good_corrections_only', action='store_true', help='Keep in the photometric loss of the flow only well-estimated corrections')
    add('--input_dim_corrector', type=int, default=6)
    add('--occ_in_correc_inputs', action='store_true', help='Include occlusions in corrector inputs')
    add('--no_FSW_occ_in_correc_inputs', action='store_true', help='Include true occlusions in corrector inputs when training with full-size warping')
    add('--occ_and_out_in_correc_inputs', action='store_true', help='Include occlusions and true boundary occlusions in corrector inputs')
    add('--flows_in_correc_inputs', action='store_true', help='Include foward flow and warped backward flow in corrector inputs')

    # GMA args
    add('--position_only', default=False, action='store_true', help='only use position-wise attention')
    add('--position_and_content', default=False, action='store_true', help='use position and content-wise attention')
    add('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')

    # SCV args
    add('--upsample-learn', action='store_true', default=False, help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    add('--gamma', type=float, default=0.8, help='exponential weighting')
    add('--num_k', type=int, default=32, help='number of hypotheses to compute for knn Faiss')
    add('--max_search_range', type=int, default=100, help='maximum search range for hypotheses in quarter resolution')
                        
    args = parser.parse_args()
    args.num_gpus = torch.cuda.device_count()
    args.batch_size //= args.num_gpus

    args.upsampling_mode = 'bilinear' if args.small else args.upsampling_mode

    set_photometric_loss_weights(args)

    if args.crop_size is None:
        if 'kitti' in args.dataset_train.lower():
            args.crop_size = [296, 696]
        elif 'chairs' in args.dataset_train.lower():
            args.crop_size = [368, 496]
        elif 'sintel' in args.dataset_train.lower():
            args.crop_size = [368, 496]
        else:
            raise NotImplementedError

    if args.occlusions is None:
        if 'kitti' in args.dataset_train.lower():
            args.occlusions = 'brox'
        elif 'chairs' in args.dataset_train.lower():
            args.occlusions = 'wang'
        elif 'sintel' in args.dataset_train.lower():
            args.occlusions = 'wang'
        else:
            raise NotImplementedError

    if args.smoothness_weight is None:
        if 'kitti' in args.dataset_train.lower():
            args.smoothness_weight = 2.
        elif 'chairs' in args.dataset_train.lower():
            args.smoothness_weight = 2.
        elif 'sintel' in args.dataset_train.lower():
            args.smoothness_weight = 2.5
        else:
            raise NotImplementedError

    if args.smoothness_order is None:
        if 'kitti' in args.dataset_train.lower():
            args.smoothness_order = 2
        elif 'chairs' in args.dataset_train.lower():
            args.smoothness_order = 1
        elif 'sintel' in args.dataset_train.lower():
            args.smoothness_order = 1
        else:
            raise NotImplementedError

    if 'correc' in args.mode:
        if args.occ_in_correc_inputs or args.no_FSW_occ_in_correc_inputs:
            args.input_dim_corrector += 1
        elif args.occ_and_out_in_correc_inputs:
            args.input_dim_corrector += 2
        if args.flows_in_correc_inputs:
            args.input_dim_corrector += 4

    return args

def assert_l1(census_weight, unflow_weight, l1_weight):
    if (unflow_weight > 0. and l1_weight > 0.) or (census_weight > 0. and l1_weight > 0):
        return False
    else:
        return True

def set_photometric_loss_weights(args):
    assert assert_l1(args.census_weight_flow, args.unflow_weight_flow, args.l1_weight_flow), 'You have to choose between L1 and Census or Unflow loss'

    if args.unflow_weight_flow > 0.:
        args.ssim_weight_flow = 0.85 * args.unflow_weight_flow
        args.l1_weight_flow = 0.15 * args.unflow_weight_flow
    else:
        args.ssim_weight_flow = 0.

    if 'correc' in args.mode:
        assert assert_l1(args.census_weight_correc, args.unflow_weight_correc, args.l1_weight_correc), 'You have to choose between L1 and Census or Unflow loss'
        if args.unflow_weight_correc > 0.:
            args.ssim_weight_correc = 0.85 * args.unflow_weight_correc
            args.l1_weight_correc = 0.15 * args.unflow_weight_correc
        else:
            args.ssim_weight_correc = 0.
        
    del args.unflow_weight_flow
    del args.unflow_weight_correc


def save_args(args, dir):
        with open(os.path.join(dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)