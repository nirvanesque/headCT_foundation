# Modified from https://github.com/cvlab-stonybrook/SelfMedMAE/blob/main/lib/models/mae3d.py and https://github.com/facebookresearch/mae

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from typing import Tuple, List, Optional

from src.utils.pos_embed import build_sincos_position_embedding
from src.utils.patch_embedding import PatchEmbeddingBlock
from src.models.attentionblock import AttentionBlock

from timm.models.layers import to_3tuple

from monai.networks.layers import trunc_normal_


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""
    def __init__(self, 
                 input_size: int,
                 patch_size: int,
                 mask_ratio: float,
                 in_chans: int = 1,
                 dropout_rate: float = 0.,
                 spatial_dims: int = 3,
                 patch_embed: str = 'conv',
                 pos_embed: str = 'learnable',
                 encoder_depth: int = 12,
                 encoder_embed_dim: int = 768,
                 encoder_mlp_dim: int = 3072,
                 encoder_num_heads: int = 12,
                 decoder_depth: int = 8,
                 decoder_embed_dim: int = 768,
                 decoder_mlp_dim: int = 3072,
                 decoder_num_heads: int = 16,
                 norm_pix_loss: bool = False,
                 use_bias: bool = False,
                 norm_layer: nn.Module = nn.LayerNorm
    ):
        """
        Initializes the MAE (Masked Autoencoder) model.
        Args:
            input_size (int): Size of the input image.
            patch_size (int): Size of the patches to be extracted from the input image.
            mask_ratio (float): Ratio of patches to be masked.
            in_chans (int, optional): Number of input channels. Default is 1.
            dropout_rate (float, optional): Dropout rate. Default is 0.
            spatial_dims (int, optional): Number of spatial dimensions. Default is 3.
            patch_embed (str, optional): Type of patch embedding. Default is 'conv'.
            pos_embed (str, optional): Type of positional embedding. Default is 'learnable'.
            encoder_depth (int, optional): Number of encoder layers. Default is 12.
            encoder_embed_dim (int, optional): Dimension of encoder embeddings. Default is 768.
            encoder_mlp_dim (int, optional): Dimension of encoder MLP. Default is 3072.
            encoder_num_heads (int, optional): Number of attention heads in the encoder. Default is 12.
            decoder_depth (int, optional): Number of decoder layers. Default is 8.
            decoder_embed_dim (int, optional): Dimension of decoder embeddings. Default is 768.
            decoder_mlp_dim (int, optional): Dimension of decoder MLP. Default is 3072.
            decoder_num_heads (int, optional): Number of attention heads in the decoder. Default is 16.
            norm_pix_loss (bool, optional): Whether to use normalized pixel loss. Default is False.
            use_bias (bool, optional): Whether to use bias in linear layers. Default is False.
            norm_layer (nn.Module, optional): Normalization layer to use. Default is nn.LayerNorm.
        """
        super().__init__()
        
        input_size = to_3tuple(input_size)
        patch_size = to_3tuple(patch_size)
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.spatial_dims = spatial_dims
        self.pos_embed = pos_embed
        self.norm_pix_loss = norm_pix_loss
        
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        out_chans = in_chans * np.prod(patch_size)
        self.out_chans = out_chans
        
        grid_size = [in_size // pa_size for in_size, pa_size in zip(input_size, patch_size)]
        self.grid_size = grid_size
        
        num_patches = np.prod(grid_size)
        patch_dim = np.prod(patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        
        self.patch_embedding = PatchEmbeddingBlock(
            img_size=input_size,
            patch_size=patch_size,
            in_channels=in_chans,
            hidden_size=encoder_embed_dim,
            num_heads=encoder_num_heads,
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        
        self.blocks = nn.ModuleList([
            AttentionBlock(encoder_embed_dim, encoder_mlp_dim, encoder_num_heads, dropout_rate, qkv_bias=use_bias, save_attn=False, norm_layer=norm_layer)
            for _ in range(encoder_depth)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            AttentionBlock(decoder_embed_dim, decoder_mlp_dim, decoder_num_heads, dropout_rate, qkv_bias=use_bias, save_attn=False, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
    
        self.norm = norm_layer(encoder_embed_dim)
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=use_bias)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim * in_chans, bias=use_bias) 
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights for the model."""
        if self.pos_embed == "sincos":
            with torch.no_grad():
                decoder_pos_embed = build_sincos_position_embedding(self.grid_size, self.decoder_embed_dim, self.spatial_dims)
                self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())
        else:
            trunc_normal_(self.decoder_pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.decoder_cls_token, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for linear and layer norm layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x: Tensor) -> Tensor:
        """
        Patchify input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W, D].
        
        Returns:
            Tensor: Patchified tensor of shape [B, gh*gw*gd, ph*pw*pd*C].
        """
        B, C, H, W, D = x.shape
        patch_size = self.patch_size
        grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])
        
        gh, gw, gd = grid_size
        ph, pw, pd = patch_size
        
        x = x.reshape(B, C, gh, ph, gw, pw, gd, pd)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size), np.prod(patch_size) * C)
        
        return x
    
    def unpatchify(self, x: Tensor, x_ori: Tensor) -> Tensor:
        """
        Unpatchify input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [B, gh*gw*gd, ph*pw*pd*C].
            x_ori (Tensor): Original input tensor of shape [B, C, H, W, D].
        
        Returns:
            Tensor: Unpatchified tensor of shape [B, C, gh*ph, gw*pw, gd*pd].
        """
        B, C, H, W, D = x_ori.shape
        patch_size = self.patch_size
        
        ph, pw, pd = patch_size
        gh, gw, gd = H // ph, W // pw, D // pd

        x = x.reshape(B, gh, gw, gd, ph, pw, pd, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, gh*ph, gw*pw, gd*pd)

        return x
    
    def random_masking(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Apply random masking to the input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [N, L, D].
        
        Returns:
            Tuple: Masked tensor, mask, ids_restore, and ids_keep.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep
    
    def forward_encoder(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tuple: Latent tensor, mask, and ids_restore.
        """
        x = self.patch_embedding(x)
        x, mask, ids_restore, ids_keep = self.random_masking(x)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        residual = None
        for blk in self.blocks:
            x, residual = blk(x, residual)
        
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x (Tensor): Latent tensor.
            ids_restore (Tensor): Indices to restore the original order.
        
        Returns:
            Tensor: Reconstructed tensor.
        """
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)
        decoder_pos_embed = self.decoder_pos_embed.expand(x.shape[0], -1, -1)
        decoder_pos_embed = torch.cat((decoder_cls_token, decoder_pos_embed), dim=1)
        x = x + decoder_pos_embed
        
        residual = None
        for blk in self.decoder_blocks:
            x, residual = blk(x, residual)
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """
        Compute the loss between the original and reconstructed images.
        
        Args:
            imgs (Tensor): Original images.
            pred (Tensor): Reconstructed images.
            mask (Tensor): Mask applied to the images.
        
        Returns:
            Tensor: Computed loss.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        mask = mask.view(loss.shape)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, x: Tensor) -> Tuple[Tensor, None, None]:
        """
        Forward pass through the entire model.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tuple: Loss, None, None.
        """
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        
        return loss, None, None