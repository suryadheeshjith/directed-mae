# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import logging

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def center_masking(self, x, mask_ratio):
        """
        Mask out the center of the image based on a fixed mask ratio.
        x: [N, L, D], where N is the batch size, L is the total number of patches, and D is the dimension of each patch.
        """
        N, L, D = x.shape  # batch, length, dim

        # Assuming square grid
        center_length = int(torch.sqrt(mask_ratio * torch.tensor(L)))
        grid_length = int(torch.sqrt(torch.tensor(L).float()))

        # Calculate start and end indices for the center
        start_idx = (grid_length - center_length) // 2
        end_idx = start_idx + center_length

        # Create a mask of zeros with the same shape as the image patches
        mask = torch.zeros([x.shape[0], grid_length, grid_length], device=x.device)

        # Set the center region of the mask to 1
        mask[:, start_idx:end_idx, start_idx:end_idx] = 1

        # Reshape the mask to [N, L]
        mask = mask.view(x.shape[0], -1)

        # Mask/Remove the center patches
        x_masked = x[mask==0].view(N,-1,D)

        ids_shuffled = torch.sort(mask, dim=1, stable=True)[1]
        ids_restore = torch.argsort(ids_shuffled, dim=1)

        return x_masked, mask, ids_restore
    
    def random_block_masking(self, x, l_mask_ratio, u_mask_ratio):
        """
        Mask out a random block of the image based on a random mask ratio between l_mask_ratio and u_mask_ratio.
        x: [N, L, D], where N is the batch size, L is the total number of patches, and D is the dimension of each patch.
        """
        N, L, D = x.shape  # batch, length, dim

        # Assuming square grid
        grid_length = int(torch.sqrt(torch.tensor(L).float()))
        
        # Randomly determine the size of the block to mask
        mask_size = torch.randint(int(l_mask_ratio * L), int(u_mask_ratio * L) + 1, (1,))
        block_length = int(torch.sqrt(mask_size))


        # Randomly determine the start index for the block
        max_start_idx = grid_length - block_length
        start_idx_row = torch.randint(0, max_start_idx + 1, (1,))
        start_idx_col = torch.randint(0, max_start_idx + 1, (1,))

        end_idx_row = start_idx_row + block_length
        end_idx_col = start_idx_col + block_length

        # Create a mask of zeros with the same shape as the image patches
        mask = torch.zeros([x.shape[0], grid_length, grid_length], device=x.device)

        # Set the block region of the mask to 1
        mask[:, start_idx_row:end_idx_row, start_idx_col:end_idx_col] = 1

        # Reshape the mask to [N, L]
        mask = mask.view(x.shape[0], -1)

        # Mask/Remove the block patches
        x_masked = x[mask==0].view(N,-1,D)

        ids_shuffled = torch.sort(mask, dim=1, stable=True)[1]
        ids_restore = torch.argsort(ids_shuffled, dim=1)

        return x_masked, mask, ids_restore
    
    def positional_block_masking(self, x, mask_ratio, x_start, y_start):
        """
        Mask a block at the given (x_start, y_start) position in the image.
        x: [N, L, D], where N is the batch size, L is the total number of patches, and D is the dimension of each patch.
        x_start, y_start: The starting coordinates for the block mask.
        """
        N, L, D = x.shape  # batch, length, dim

        # Assuming square grid
        grid_length = int(torch.sqrt(torch.tensor(L).float()))
        
        # Determine the size of the block to mask
        mask_size = int(mask_ratio * L)
        block_length = int(torch.sqrt(torch.tensor(mask_size)))
        
        # Ensure the block doesn't go out of bounds
        if x_start + block_length > grid_length or y_start + block_length > grid_length:
            raise ValueError("The block mask goes out of the image boundaries!")

        # Create a mask of zeros with the same shape as the image patches
        mask = torch.zeros([x.shape[0], grid_length, grid_length], device=x.device)

        # Set the block region of the mask to 1
        mask[:, y_start:y_start+block_length, x_start:x_start+block_length] = 1

        # Reshape the mask to [N, L]
        mask = mask.view(x.shape[0], -1)

        # Mask/Remove the block patches
        x_masked = x[mask==0].view(N,-1,D)

        ids_shuffled = torch.sort(mask, dim=1, stable=True)[1]
        ids_restore = torch.argsort(ids_shuffled, dim=1)

        return x_masked, mask, ids_restore


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, l_mask_ratio, u_mask_ratio, mask_type, position_x, position_y):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio    
        if mask_type == "center":
            x, mask, ids_restore = self.center_masking(x, u_mask_ratio)
        
        elif mask_type == "random":
            x, mask, ids_restore = self.random_masking(x, u_mask_ratio)

        elif mask_type == "random_block":
            x, mask, ids_restore = self.random_block_masking(x, l_mask_ratio, u_mask_ratio)
        
        elif mask_type == "positional_block":
            x, mask, ids_restore = self.positional_block_masking(x, u_mask_ratio, position_x, position_y)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, per_image_loss):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if per_image_loss:
            loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, l_mask_ratio=0.4, u_mask_ratio=0.75, mask_type="random", per_image_loss=False, position_block=(0,0)):
        latent, mask, ids_restore = self.forward_encoder(imgs, l_mask_ratio, u_mask_ratio, mask_type, position_block[0], position_block[1])
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, per_image_loss)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
