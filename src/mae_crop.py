# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------



import omegaconf
import os
import time
import json

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.misc as misc
import utils

import models_mae

from engine_pretrain import get_crops

import timm
assert timm.__version__ == "0.3.2"  # version check

def main(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    print("{}".format(resolved_args).replace(', ', ',\n'))

    cudnn.benchmark = True
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    model_without_ddp = model.module
    misc.load_model(args, model_without_ddp)
    print("Model = %s" % str(model_without_ddp))
   
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = utils.IndexImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
    
    if args.limit > 0:
        dataset = torch.utils.data.random_split(dataset, (args.limit, len(dataset)-args.limit), generator=torch.Generator().manual_seed(args.seed))[0]

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True, seed=args.seed)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print("{} train imgs ready!".format(len(dataset)))

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'GET CROPS:'
    
    data_loader.sampler.set_epoch(0)
    crop_coords = {}
    for data_iter_step, (samples, _, indices, paths) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.cuda(non_blocking=True)

        args.update({'mask_type': 'random_block', 
                    'num_random_masks': 5, 
                    'l_mask_ratio': args.global_crops_scale[0], 
                    'u_mask_ratio': args.global_crops_scale[1], 
                    'top_k': 2})

        global_masks = get_crops(model, samples, model_without_ddp, args=args)
        

        args.update({'mask_type': 'random_block', 
                    'num_random_masks': args.local_crops_number*2, 
                    'l_mask_ratio': args.local_crops_scale[0], 
                    'u_mask_ratio': args.local_crops_scale[1], 
                    'top_k': args.local_crops_number})
    
        local_masks = get_crops(model, samples, model_without_ddp, args=args)
        global_crops_coords = utils.get_mask_coords(global_masks)
        local_crops_coords = utils.get_mask_coords(local_masks)
        for i, index in enumerate(indices):
            crop_coords[index.item()] = {'path': paths[i], 'local_crops': local_crops_coords[i], 'global_crops': global_crops_coords[i]}

    torch.save(crop_coords, os.path.join(args.output_dir, 'crop_coords.pth'))


    