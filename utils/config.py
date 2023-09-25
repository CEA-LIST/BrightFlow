import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import random

import datasets
import augmentations
import models
import runners
from utils.metrics import Metrics


def configure_random_seed(seed, gpu=0):
    '''Set seeds'''
    seed = seed + gpu

    # python
    random.seed(seed)

    # numpy
    seed += 1
    np.random.seed(seed)

    # torch
    seed += 1
    torch.manual_seed(seed)

    # torch cuda
    seed += 1
    torch.cuda.manual_seed(seed)


def get_train_dataloaders(args, rank=None):
    '''Set train dataloader'''
    if args.mode == 'flow_correc':
        augmentor = augmentations.AugmentorCorrections
    elif args.mode == 'flow_only':
        augmentor = augmentations.Augmentor
    else:
        raise NotImplementedError

    train_dataset = getattr(datasets, args.dataset_train)(args, augmentor, is_training=True, split='training')
    if rank == None:
        train_loader = data.DataLoader(train_dataset, 
                                       batch_size=args.batch_size, 
                                       pin_memory=True, 
                                       shuffle=True, 
                                       num_workers=args.num_workers, 
                                       drop_last=True)
    else:
        train_loader = data.DataLoader(train_dataset, 
                                       batch_size=args.batch_size, 
                                       pin_memory=True, 
                                       shuffle=False, 
                                       num_workers=args.num_workers, 
                                       drop_last=True,
                                       sampler=torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.num_gpus, rank=rank))
    return train_loader

def get_test_dataloaders(args, rank=None):
    '''Set test dataloader'''
    test_dataset = getattr(datasets, args.dataset_test)(args, augmentor=None, is_training=False, split='training' if args.eval_on_train else 'validation')

    if rank == None:
        test_loader = data.DataLoader(test_dataset, 
                                      batch_size=1, 
                                      pin_memory=True, 
                                      shuffle=False, 
                                      num_workers=2, 
                                      drop_last=False)
    else:
        test_loader = data.DataLoader(test_dataset, 
                                      batch_size=1, 
                                      pin_memory=True, 
                                      shuffle=False, 
                                      num_workers=2, 
                                      drop_last=False,
                                      sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=args.num_gpus, rank=rank, shuffle=False))
    return test_loader

def get_runner(args):
    '''Set runner (model(s) + losses)'''
    model = getattr(models, args.model)
    if args.mode == 'flow_correc':
        runner = runners.RunnerCorrection(model, args)
    elif args.mode == 'flow_only':
        runner = runners.Runner(model, args)
    else:
        raise NotImplementedError
    return runner

def get_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError

    if args.scheduler == 'smurf':
        # SMURF scheduler with optional warmup
        lambda_lr = lambda step: (0.001 + step/args.end_warmup_step if step < args.end_warmup_step else 1) * \
                                (0.5 ** ((step + args.init_step - (args.num_steps - args.lr_decay_step)) / (np.log(0.5)/np.log(args.lr_decay_max)*args.lr_decay_step)) if step + args.init_step > (args.num_steps - args.lr_decay_step) else 1.)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif args.scheduler == 'raft':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    else:
        raise NotImplementedError

    return optimizer, scheduler
    

def get_metrics(args):
    '''Set evaluator'''
    return Metrics(args, args.dataset_test)