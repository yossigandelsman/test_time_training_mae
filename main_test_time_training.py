# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import signal

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm
from torchvision import datasets
import glob
import util.misc as misc
import models_mae_shared
from engine_test_time import train_on_test, get_prameters_from_args
from data import tt_image_folder
from util.misc import NativeScalerWithGradNormCount as NativeScaler



def get_args_parser():
    parser = argparse.ArgumentParser('MAE test time training', add_help=False)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--finetune_mode', default='encoder', type=str, help='all, encoder, encoder_no_cls_no_msk.')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--classifier_depth', type=int, metavar='N', default=0,
                        help='number of blocks in the classifier')
    # Test time training
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--steps_per_example', default=1, type=int,)
    parser.add_argument('--stored_latents', default='', help='have we generated the latents already?')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # Dataset parameters
    parser.add_argument('--batch_size', default=256, type=int,)
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--dataset_name', default='imagenet_c', type=str,
                        help='dataset name')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--load_loss_scalar', action='store_true')
    parser.set_defaults(load_loss_scalar=False)
    parser.add_argument('--optimizer_type', default='sgd', help='adam, adam_w, sgd.')
    parser.add_argument('--optimizer_momentum', default=0.9, type=float, help='adam, adam_w, sgd.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume_model', default='', required=True, help='resume from checkpoint')
    parser.add_argument('--resume_finetune', default='', required=True, help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--head_type', default='vit_head',
                        help='Head type - linear or vit_head')
    
    parser.add_argument('--single_crop', action='store_true',
                        help='single_crop training')
    parser.add_argument('--no_single_crop', action='store_false', dest='single_crop')
    parser.set_defaults(single_crop=False)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def load_combined_model(args, num_classes: int = 1000):
    if args.model == 'mae_vit_small_patch16':
        classifier_depth = 8
        classifier_embed_dim = 512
        classifier_num_heads = 16
    else:
        assert 'mae_vit_large_patch16' in args.model or 'mae_vit_huge_patch14' in args.model
        classifier_embed_dim = 768
        classifier_depth = 12
        classifier_num_heads = 12
    model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type, norm_pix_loss=args.norm_pix_loss, 
                                                   classifier_depth=classifier_depth, classifier_embed_dim=classifier_embed_dim, 
                                                   classifier_num_heads=classifier_num_heads,
                                                   rotation_prediction=False)
    model_checkpoint = torch.load(args.resume_model, map_location='cpu')
    head_checkpoint = torch.load(args.resume_finetune, map_location='cpu')
    
    if args.head_type == 'linear':
        model_checkpoint['model']['bn.running_mean'] = head_checkpoint['model']['head.0.running_mean']
        model_checkpoint['model']['bn.running_var'] = head_checkpoint['model']['head.0.running_var']
        model_checkpoint['model']['head.weight'] = head_checkpoint['model']['head.1.weight']
        model_checkpoint['model']['head.bias'] = head_checkpoint['model']['head.1.bias']
    else:
        assert args.classifier_depth != 0, 'Please provide classifier_depth parameter.'
        for key in head_checkpoint['model']:
            if key.startswith('classifier'):
                model_checkpoint['model'][key] = head_checkpoint['model'][key]
    model.load_state_dict(model_checkpoint['model'])
    optimizer = None
    if args.load_loss_scalar:
        loss_scaler = NativeScaler()
        loss_scaler.load_state_dict(model_checkpoint['scaler'])
    else:
        loss_scaler = None
    return model, optimizer, loss_scaler

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    max_known_file = max([int(i.split('results_')[-1].split('.npy')[0]) for i in glob.glob(os.path.join(args.output_dir, 'results_*.npy'))] + [-1])
    if max_known_file != -1:
        print(f'Found {max_known_file} values, continues from next iterations.')
        
    # simple augmentation    
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if not args.single_crop:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    data_path = args.data_path

    dataset_train = tt_image_folder.ExtendedImageFolder(data_path, transform=transform_train, minimizer=None, 
                                                        batch_size=args.batch_size, steps_per_example=args.steps_per_example * args.accum_iter, 
                                                        single_crop=args.single_crop, start_index=max_known_file+1)

    dataset_val = tt_image_folder.ExtendedImageFolder(data_path, transform=transform_val, 
                                                        batch_size=1, minimizer=None, 
                                                        single_crop=args.single_crop, start_index=max_known_file+1)

    num_classes = 1000

    # define the model
    model, optimizer, scalar = load_combined_model(args, num_classes)
         
    print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    args.lr = args.blr * eff_batch_size / 256

    wandb_config = vars(args)
    base_lr = (args.lr * 256 / eff_batch_size)
    wandb_config['base_lr'] = base_lr
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    start_time = time.time()
    test_stats = train_on_test(
        model, optimizer, scalar, dataset_train, dataset_val,
        device,
        log_writer=None,
        args=args,
        num_classes=num_classes,
        iter_start=max_known_file+1
    )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
