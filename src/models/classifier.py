from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

class LinearClassifier(nn.Module):
    """
    A simple linear classifier with batch normalization.
    """
    def __init__(self, dim: int, num_classes: int):
        """
        Args:
            dim (int): Dimension of the input features.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.bn(x)
        x = self.linear(x)
        return x
    
class AttentionClassifier(nn.Module):
    """
    A classifier with attention mechanism.
    """
    def __init__(
        self, 
        dim: int, 
        num_classes: int, 
        num_heads: int = 12, 
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 1,
    ):
        """
        Args:
            dim (int): Dimension of the input features.
            num_classes (int): Number of output classes.
            num_heads (int): Number of attention heads. Default is 12.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default is False.
            qk_scale (Optional[float]): Override default qk scale of head_dim ** -0.5 if set.
            num_queries (int): Number of query tokens. Default is 1.
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_queries = num_queries
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.bn1 = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        self.bn2 = nn.BatchNorm1d(dim, affine=False, eps=1e-6)

        self.wkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.linear = nn.Linear(dim, num_classes)
        
        self.cls_token = nn.Parameter(torch.zeros(1, num_queries, dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        q = cls_tokens.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        x = self.bn1(x.transpose(-2, -1)).transpose(-2, -1)
        kv = self.wkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        x_cls = attn_out.reshape(B, self.num_queries, C)
        x_cls = self.bn2(x_cls.transpose(-2, -1)).transpose(-2, -1)
        x_cls = x_cls.mean(dim=1)
        
        out = self.linear(x_cls)
        return out