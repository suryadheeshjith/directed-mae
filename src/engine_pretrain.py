# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import logging

import torch

import util.misc as misc
import util.lr_sched as lr_sched
# from util.wandb import get_wandb_image, get_wandb_plot
# import matplotlib.pyplot as plt
from util.image import unnormalize_image


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, model_without_ddp, 
                    log_writer=None,
                    args=None, wandb=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        logging.info('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if args.limit != -1:
            if args.batch_size * data_iter_step > args.limit:
                break

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, l_mask_ratio=args.l_mask_ratio, u_mask_ratio=args.u_mask_ratio, mask_type=args.mask_type)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logging.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        

        if wandb:
            wandb.log({"train_loss_per_batch": loss_value_reduce, "lr_per_batch": lr})
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module,
            data_loader: Iterable,
            device: torch.device, model_without_ddp, 
            args=None, wandb=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    loss_full_list = []

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        logging.info("Running step {}".format(data_iter_step))
        if args.limit > 0:
            if args.batch_size * data_iter_step > args.limit:
                break

        samples = samples.to(device, non_blocking=True)

        if args.mask_type == "positional_block":
            if args.batch_size != 1:
                raise ValueError("Batch Size must be 1 for positional block masking")
            
            N, L, D = model.patch_embed(samples).shape
            grid_length = int(torch.sqrt(torch.tensor(L).float()))
            
            mask_size = int(args.u_mask_ratio * L)
            block_length = int(torch.sqrt(torch.tensor(mask_size)))

            new_samples = torch.clone(samples) # Single sample
            loss = None
            y = None
            mask = None
            loss_list = []
            
            for start_idx_row in range(0, grid_length - block_length + 1, args.mask_stride):
                for start_idx_col in range(0, grid_length - block_length + 1, args.mask_stride):
                    with torch.cuda.amp.autocast():
                        new_loss, new_y, new_mask = model(new_samples, l_mask_ratio=args.l_mask_ratio, u_mask_ratio=args.u_mask_ratio,\
                                                         mask_type=args.mask_type, per_image_loss=False, position_block=(start_idx_col,start_idx_row))
                    
                    if loss is None:
                        loss = torch.tensor([new_loss])
                        y = new_y
                        mask = new_mask
                    
                    else:
                        loss = torch.cat([loss, torch.tensor([new_loss])], dim=0)
                        y = torch.cat([y, new_y], dim=0)
                        mask = torch.cat([mask, new_mask], dim=0)
                        samples = torch.cat([samples, new_samples], dim=0)

                    if wandb:
                        loss_list.append(new_loss.item())
            
            if wandb:
                # Normalizing losses across all images, all mask positions
                # loss_full_list += loss_list

                # Normalizing losses across each image, for all positions
                loss_full_list.append(loss_list)

            if args.limit != -1 and args.limit < 500 and args.batch_size == 1 and data_iter_step % args.pos_log_period == 0 and wandb:                
                plt.close()
                fig = plt.figure()
                plt.plot(loss_list)
                plt.xlabel("Patch")
                plt.ylabel("Loss")
                wandb.log({"image_{:04d}_loss".format(data_iter_step) : get_wandb_image(plt, mode="plt")})

                # Plot norm loss

                loss_tensor = torch.tensor(loss_list)

                mean = loss_tensor.mean(dim=-1, keepdim=True)
                var = loss_tensor.var(dim=-1, keepdim=True)
                loss_tensor = (loss_tensor - mean) / (var + 1.e-6)**.5

                plt.close()
                fig = plt.figure()
                plt.plot(loss_tensor)
                plt.xlabel("Patch")
                plt.ylabel("Norm Loss")
                wandb.log({"image_{:04d}_norm_loss".format(data_iter_step) : get_wandb_image(plt, mode="plt")})

            loss_value = loss.mean().item()
            
        else:

            with torch.cuda.amp.autocast():
                loss, y, mask = model(samples, l_mask_ratio=args.l_mask_ratio, u_mask_ratio=args.u_mask_ratio, mask_type=args.mask_type)

            loss_value = loss.item()
            metric_logger.update(loss=loss.item())

            if not math.isfinite(loss_value):
                logging.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            if args.duplicate_comparison > 0 and args.batch_size==1:
                new_samples = torch.clone(samples) # Single sample
                loss = torch.tensor([loss])
                for i in range(args.duplicate_comparison):
                    new_loss, new_y, new_mask = model(new_samples, l_mask_ratio=args.l_mask_ratio, u_mask_ratio=args.u_mask_ratio, mask_type=args.mask_type)
                    
                    loss = torch.cat([loss, torch.tensor([new_loss])], dim=0)
                    y = torch.cat([y, new_y], dim=0)
                    mask = torch.cat([mask, new_mask], dim=0)
                    samples = torch.cat([samples, new_samples], dim=0)
                
                loss_value = loss.mean().item()

        # loss_value_reduce = misc.all_reduce_mean(loss_value)

        if wandb:
            # wandb.log({"eval_loss_per_batch": loss_value_reduce})
            wandb.log({"eval_loss_per_batch": loss_value})
        
        
        if args.limit != -1 and args.limit < 500 and args.batch_size == 1 and data_iter_step % args.pos_log_period == 0 and wandb:
            x = samples
            y = model_without_ddp.unpatchify(y)

            mask = mask.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_embed.patch_size[0]**2 *3)
            mask = model_without_ddp.unpatchify(mask)
            im_masked = x * (1 - mask)
            im_paste = x * (1 - mask) + y * mask

            for i in range(x.shape[0]):

                org_image = unnormalize_image(x[i])
                masked_image = unnormalize_image(im_masked[i])
                impainted_image = unnormalize_image(im_paste[i])

                wandb_images = []
                wandb_images.append(get_wandb_image(org_image, caption="Original"))
                wandb_images.append(get_wandb_image(masked_image, caption="Masked"))
                wandb_images.append(get_wandb_image(impainted_image, caption="Predicted"))

                wandb.log({"image_{:04d}_{:04d} loss = {}".format(data_iter_step, i, loss[i].item()): wandb_images})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Loss histogram
    if wandb:
        plt.close()
        fig = plt.figure()
        loss_full_tensor = torch.tensor(loss_full_list)

        # Save loss_full_tensor
        torch.save(loss_full_tensor, args.output_dir + '/loss_full_tensor.pt')

        mean = loss_full_tensor.mean(dim=-1, keepdim=True)
        var = loss_full_tensor.var(dim=-1, keepdim=True)
        loss_full_tensor = (loss_full_tensor - mean) / (var + 1.e-6)**.5

        # Save normalized loss_full_tensor
        torch.save(loss_full_tensor, args.output_dir + '/norm_loss_full_tensor.pt')

        loss_full_tensor = loss_full_tensor.reshape((-1,))
        
        plt.hist(loss_full_tensor, bins=50)
        plt.xlabel("Normalized Loss")
        wandb.log({"Loss histogram (Mask ratio: {}, Stride: {})".format(args.u_mask_ratio, args.mask_stride) : get_wandb_image(plt, mode="plt")})
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def get_crops(model: torch.nn.Module,
            samples,
            model_without_ddp, 
            args=None, wandb=None):
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()

    N, L, D = model_without_ddp.patch_embed(samples).shape

    loss = None
    y = None
    mask = None

    if args.mask_type == "positional_block":     
        grid_length = int(torch.sqrt(torch.tensor(L).float()))
        mask_size = int(args.u_mask_ratio * L)
        block_length = int(torch.sqrt(torch.tensor(mask_size)))
        
        num_candidates = len(range(0, grid_length - block_length + 1, args.mask_stride))**2   
        if args.top_k > num_candidates:
            raise ValueError("top_k too high, must be lower than {}".format(num_candidates))

        for start_idx_row in range(0, grid_length - block_length + 1, args.mask_stride):
            for start_idx_col in range(0, grid_length - block_length + 1, args.mask_stride):
                with torch.cuda.amp.autocast():
                    new_loss, _, new_mask = model(samples, l_mask_ratio=-1, u_mask_ratio=args.u_mask_ratio,\
                                                    mask_type=args.mask_type, per_image_loss=True, position_block=(start_idx_col,start_idx_row))
                
                
                new_loss = new_loss.unsqueeze(1)
                new_mask = new_mask.unsqueeze(1)

                if loss is None:
                    loss = new_loss
                    mask = new_mask
                
                else:
                    loss = torch.cat([loss, new_loss], dim=1)
                    mask = torch.cat([mask, new_mask], dim=1)

        
    elif args.mask_type == "random_block":   
        if args.top_k > args.num_random_masks:
            raise ValueError("top_k too high, must be lower than {}".format(num_random_masks))

        for i in range(args.num_random_masks):
            with torch.cuda.amp.autocast():
                new_loss, _, new_mask = model(samples, l_mask_ratio=args.l_mask_ratio, u_mask_ratio=args.u_mask_ratio, mask_type=args.mask_type, per_image_loss=True)
            
            new_loss = new_loss.unsqueeze(1)
            new_mask = new_mask.unsqueeze(1)

            if loss is None:
                loss = new_loss
                mask = new_mask
            
            else:
                loss = torch.cat([loss, new_loss], dim=1)
                mask = torch.cat([mask, new_mask], dim=1)
    
    else:
        print("Enter valid mask type")
        exit(0)

    topk_loss_indx = loss.topk(args.top_k, dim=1)[1] # N, top_k
    topk_losses = loss.gather(dim=1, index=topk_loss_indx) # N, top_k
    topk_masks = mask.gather(dim=1, index=topk_loss_indx.unsqueeze(-1).repeat(1,1,L)) # N, top_k, num_patches

    topk_masks = topk_masks.reshape(N*args.top_k, -1) # N x top_k, num_patches
    topk_masks = topk_masks.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_embed.patch_size[0]**2 *3) # N x top_k, num_patches, num_pixel_values_per_patch [16**2 *3 (768)]
    topk_masks = model_without_ddp.unpatchify(topk_masks) # N x top_k, C, H, W
    size = topk_masks.shape[-1]
    topk_masks = topk_masks.reshape(3, N, args.top_k, size, size) # C, N, top_k, H, W

    topk_masks = topk_masks[0] # Redundant channel dim = N, top_k, H, W
    
    # For reference:
    # im_masked = samples * (1 - topk_masks)
    # im_paste = samples * (1 - topk_masks) + y * topk_masks

    # return samples * topk_masks # This would be the correct crop to provide to SSL. This will not work because need to resample samples to match masks.

    return topk_masks