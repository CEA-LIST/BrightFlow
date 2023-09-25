import os
import numpy as np
import torch
import torch.nn as nn

import utils.argument_parser as argument_parser
import utils.config as cfg
from utils.utils import to_cuda, Logging, stack_all_gather_without_backprop, CheckpointSavior
from evaluate import Validation

import torch.multiprocessing as mp
    

def train(gpu, args):
    print(gpu)

    # set random seeds
    cfg.configure_random_seed(args.seed, gpu=gpu)

    args.gpu = gpu

    torch.distributed.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.num_gpus,
    	rank=gpu
    )

    ## get dataloaders
    train_loader = cfg.get_train_dataloaders(args, gpu)
    test_loader = cfg.get_test_dataloaders(args, gpu)

    ## get runner and loss
    runner = cfg.get_runner(args)

    torch.cuda.set_device(gpu)
    runner.cuda(gpu)
    runner = nn.parallel.DistributedDataParallel(runner, device_ids=[gpu], find_unused_parameters=('correc' in args.mode and args.init_step < args.correc_starting_step))
    runner.train()

    if args.restore_ckpt is not None:
        missing_keys, unexpected_keys = runner.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        if gpu == 0:
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)

    ## get metrics
    metrics = cfg.get_metrics(args)

    ## Init validator
    validator = Validation(args)

    ## get optimizer
    optimizer, scheduler = cfg.get_optimizer(args, runner)

    save_checkpoint = CheckpointSavior(args)
    logger = Logging(runner, scheduler, args)
    
    total_steps = args.init_step
    prev_loss_dict = {}
    should_keep_training = True
    while should_keep_training:

        for _, example in enumerate(train_loader):

            ## Reset gradients
            runner.zero_grad(set_to_none=True)

            ## Transfer to cuda
            to_cuda(example)

            ## Run forwad pass
            _, loss_dict = runner(example, total_steps)

            ## Check total_loss for NaNs
            training_loss = loss_dict['loss_total']
            if np.isnan(training_loss.item()):
                print('Current loss dict:', gpu, loss_dict)
                print()
                print('Previous loss dict:', gpu, prev_loss_dict)
                print()
                raise ValueError("training_loss is NaN")
            else:
                prev_loss_dict = loss_dict

            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(runner.parameters(), args.clip)
            
            # else:
            optimizer.step()
            scheduler.step()

            ## increment total step
            total_steps += 1

            for key, value in loss_dict.items():
                loss_dict[key] = stack_all_gather_without_backprop(value).mean()
            logger.push(loss_dict)

            if total_steps % args.VAL_FREQ == args.VAL_FREQ - 1:
                runner.eval()
                results = validator.validate(runner, test_loader, metrics, total_steps)
                runner.train()

                logger.write_dict(results)
                save_checkpoint(results, runner)

            if total_steps >= args.num_steps:
                should_keep_training = False
                save_checkpoint(results, runner)
                break


if __name__ == '__main__':

    ## get arguments
    args = argument_parser.get_arguments()
    print(args)

    ## get checkpoints
    if not args.debug:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        log_dir = os.path.join(args.ckpt_dir, args.name)
        os.makedirs(log_dir, exist_ok=True)
        argument_parser.save_args(args, log_dir)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12000 + np.round(np.random.random() * 1000).astype(int))

    mp.spawn(train, nprocs=args.num_gpus, args=(args,))