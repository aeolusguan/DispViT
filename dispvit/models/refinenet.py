import logging
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
import einops

from ..layers import Mlp, DropPath
from .dispvit import InputPadder

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def fourier_coord_embed(coord, N_freqs, normalizer=3.14/512, logscale=True):
    """
    coord: [...]D
    returns:
        [...]dim, where dim=(2*N_freqs+1)*D
    """
    if logscale:
        freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs, device=coord.device)
    else:
        freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    coord = coord.unsqueeze(-1) * normalizer
    freq_bands = freq_bands[(*((None,) * (len(coord.shape) - 1)), Ellipsis)]
    f_coord = coord * freq_bands
    embed = torch.cat([f_coord.sin(), f_coord.cos(), coord], dim=-1)
    embed = einops.rearrange(embed, '... d n -> ... (d n)')

    return embed


def window_partition(x, window_size):
    """
    x: [B,H,W,C]
    Returns:
        (num_windows*B,window_size*window_size,C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B,window_size,window_size,C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head positional sensitive self attention (W-MSA).
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        shift_size (int): Shift size for SW-MSA.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override a default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, qkv_dim, window_size, shift_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(qkv_dim, dim * 3, bias=qkv_bias)

        # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # define a parameter table of relative position bias
        self.relative_position_enc_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), dim*3))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_enc_table, std=.02)

    @staticmethod
    def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
        """
        input_resolution (tuple[int]): The height and width of input
        window_size (tuple[int]): The height, width of window
        shift_size (int): Shift size for SW-MSA.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, 'b (h hs) (w ws) c -> (b h w) (hs ws) c', hs=window_size[0], ws=window_size[1])
        mask_windows = mask_windows.squeeze(-1)  # [num_windows, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float('0.0'))
        return attn_mask

    def forward(self, x, attn_mask):
        """
        x:     [num_windows*B,window_size*window_size,C']
        mask:  [num_windows,window_size*window_size,window_size*window_size]
        Returns:
            [num_windows*B,window_size*window_size,C]
        """
        B_, L, _ = x.shape
        C = self.dim
        
        qkv = (
            self.qkv(x)
            .reshape(B_, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # positional embedding
        rpe = self.relative_position_enc_table[self.relative_position_index.view(-1)].view(
            L, L, self.num_heads, -1)
        q_rpe, k_rpe, v_rpe = rpe.chunk(3, dim=-1)

        # window attention
        q = q * self.scale
        q_rpe = q_rpe * self.scale
        qk = (q @ k.transpose(-2, -1))  # B head L C @ B head C L --> B head L L
        qr = torch.einsum('bhic,ijhc->bhij', q, k_rpe)
        kr = torch.einsum('bhjc,ijhc->bhij', k, q_rpe)
        attn = qk + qr + kr
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, L, L) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = attn.softmax(dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v + torch.einsum('bhij,ijhc->bhic', attn, v_rpe)
        x = x.transpose(1, 2).reshape(B_, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, shift_size={self.shift_size}, num_heads={self.num_heads}'
    

class CrossAttention(nn.Module):
    """ Window based multi-head positional sensitive cross attention (W-MSA).

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the query window.
        stride (int): The downsample stride of query with respect to key/value.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override a default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(
        self, 
        dim,
        q_dim,
        kv_dim,
        window_size, 
        num_heads, 
        qkv_bias=True,
        proj_bias=True,
        qk_scale=None, 
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # define a parameter table of relative position bias
        self.relative_position_enc_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), dim*3))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.q = nn.Linear(q_dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.relative_position_enc_table, std=0.02)
        
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def window_partition(self, x):
        """
        x: [B,H,W,C]
        Returns:
            [B*num_windows,num_heads,window_size*window_size,head_dim]
        """
        x = einops.rearrange(x, 'b (i hs) (j ws) (h d) -> (b i j) h (hs ws) d',
                             hs=self.window_size[0], ws=self.window_size[1], h=self.num_heads)
        return x

    def forward(self, x, context):
        """
        x:   [B,H,W,C]
        context:  [B,H,W,C]
        Returns:
            B,H,W,C
        """
        _, H, W, _ = x.shape
        q = self.q(x)
        k, v = self.kv(context).chunk(2, dim=-1)

        # pad feature maps to multiples of window size
        window_size = self.window_size
        pad_l = pad_t = 0
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        Hp = H + pad_b
        Wp = W + pad_r
        if pad_r > 0 or pad_b > 0:
            q = nn.functional.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b))
            k = nn.functional.pad(k, (0, 0, pad_l, pad_r, pad_t, pad_b))
            v = nn.functional.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))

        q = self.window_partition(q)  # [B*num_windows,num_heads,window_size*window_size,head_dim]
        k = self.window_partition(k)
        v = self.window_partition(v)

        # positional embedding
        L = q.shape[2]
        rpe = self.relative_position_enc_table[self.relative_position_index.view(-1)].view(
            L, L, self.num_heads, -1)
        q_rpe, k_rpe, v_rpe = rpe.chunk(3, dim=-1)

        # window attention
        q = q * self.scale
        q_rpe = q_rpe * self.scale
        qk = (q @ k.transpose(-2, -1))  # B head N C @ B head C N' --> B head N N'
        qr = torch.einsum('bhic,ijhc->bhij', q, k_rpe)
        kr = torch.einsum('bhjc,ijhc->bhij', k, q_rpe)
        attn = qk + qr + kr
        attn = attn.softmax(dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v + torch.einsum('bhij,ijhc->bhic', attn, v_rpe)
        x = einops.rearrange(x, '(b i j) h (hs ws) d -> b (i hs) (j ws) (h d)', 
                             i=Hp//window_size[0],
                             j=Wp//window_size[1], 
                             hs=window_size[0], 
                             ws=window_size[1])
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W].contiguous()
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def to_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    
    assert isinstance(x, int)
    return (x, x)


class SwinNMP(nn.Module):
    """Swin Message Passing Block.

    Args:
        dim (int): Number of input channels.
        qkv_dim (int): Number of input token channels
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, dim, qkv_dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, qkv_dim, window_size=to_2tuple(self.window_size), shift_size=shift_size, num_heads=num_heads,
            qk_scale=qk_scale, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def add_pos_embed(self, x, pos_embed):
        x = self.norm1(x)
        # concat latent embedding with position embedding
        x = torch.cat((x, pos_embed), dim=-1)
        return x
    
    def forward(self, x, pos_embed, attn_mask):
        """
        x: [B,H,W,C], hypothesis embedding
        pos_embed: [B,H,W,C'], encoding of the underlying disparity
        attn_mask: [num_windows, window_size*window_size, window_size*window_size],
            attention mask for SW-MSA
        Returns: [B,H,W,C]
        """
        H, W, C = x.shape[1:]
        shortcut = x
        x = self.add_pos_embed(x, pos_embed)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B,window_size*window_size,C'

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, qkv_dim={self.qkv_dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}"
    

class CrossBlock(nn.Module):
    def __init__(
        self,
        dim,
        q_dim,
        kv_dim,
        window_size,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
    ) -> None:
        super().__init__()
        self.attn = CrossAttention(
            dim, q_dim, kv_dim, to_2tuple(window_size), num_heads, qkv_bias, proj_bias, qk_scale, attn_drop, drop,
        )
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, context, x_pos=None, context_pos=None):
        shortcut = x
        if x_pos is not None:
            x = torch.cat((self.norm1(x), x_pos), dim=-1)
        else:
            x = self.norm1(x)
        if context_pos is not None:
            context = torch.cat((self.norm_y(context), context_pos), dim=-1)
        else:
            context = self.norm_y(context)
        x = shortcut + self.drop_path(self.attn(x, context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class RefineLayer(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        shift_size,
        num_heads,
        mlp_ratio=4.0,
        activation="gelu",
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
    ):
        super().__init__()

        act_layer = _get_activation_fn(activation)
        qkv_dim = dim + 31 
        self.window_size = window_size
        self.shift_size = shift_size
        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.nmp = SwinNMP(dim, qkv_dim=qkv_dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size, mlp_ratio=mlp_ratio, 
                           attn_drop=attn_drop, drop_path=drop_path, drop=drop, act_layer=act_layer, norm_layer=norm_layer)
        
        kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_mem=True,
        )
        self.readout = CrossBlock(q_dim=dim+31, kv_dim=dim, window_size=window_size, **kwargs)
        self.update = CrossBlock(q_dim=dim, kv_dim=dim+31, window_size=window_size, **kwargs)
        
    def forward(self, x, pos_embed, attn_mask, context):
        """
        x: [B,H,W,C], hypothesis embedding
        pos_embed: [B,H,W,C'], encoding of the underlying disparity
        attn_mask: [num_windows,window_size*window_size,window_size*window_size],
            attention mask for SW-MSA
        Returns: [B,H,W,C]
        """
        # readout
        x = self.readout(x, context, x_pos=pos_embed)
        x = self.nmp(x, pos_embed=pos_embed, attn_mask=attn_mask)
        # update
        context = self.update(context, x, context_pos=pos_embed)
        return x, context


class Head(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class RefineNet(nn.Module):
    def __init__(
        self,
        regressor,
        regressor_pretrained_path,
        extractor,
        embed_dim,
        num_blocks,
        window_size,
        num_heads,
        mlp_ratio=4.0,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        return_intermediate=True,
        **kwargs_ignored,
    ):
        super().__init__()

        self.regressor_cfg = regressor
        self.extractor_cfg = extractor
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.return_intermediate = return_intermediate
        self.use_reentrant = False # hardcoded to False

        self.padder = None

        feat_dim = self.extractor_cfg.output_dim
        self.concatconv = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        )
        self.gw = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False)
        )

        # self.feature_down = nn.Conv2d(128, embed_dim, kernel_size=4, stride=4, padding=0, bias=False)
        self.ffn = Mlp(embed_dim+32, embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

        dpr = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]
        self.blocks = nn.ModuleList([
            RefineLayer(
                dim=embed_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            ) for i in range(num_blocks)
        ])

        self.head = Head(embed_dim, embed_dim, 4*4, 3)
        self.vit_refine = Head(embed_dim, embed_dim, 4*4, 3)

        # init weights
        self.apply(self._init_weights)

        self.extractor = instantiate(self.extractor_cfg)
        self.regressor = instantiate(self.regressor_cfg)

        # Load regressor model state
        if regressor_pretrained_path is not None:
            logging.info(f"Resuming regression model from {regressor_pretrained_path}")
            with g_pathmgr.open(regressor_pretrained_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            missing, unexpected = self.regressor.load_state_dict(
                model_state_dict, strict=True,
            )
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load sceneflow pretrained model
        # logging.info(f"Resuming refinenet model from logs/refinenet_rvc/ckpts/checkpoint.pt")
        # with g_pathmgr.open("logs/refinenet_rvc/ckpts/checkpoint.pt", "rb") as f:
        #     checkpoint = torch.load(f, map_location="cpu")
        # model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        # missing, unexpected = self.load_state_dict(
        #     model_state_dict, strict=True,
        # )
        # logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Freeze regressor weights
        # self.free_regressor()

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def free_regressor(self):
        def _freeze_model(model):
            model = model.eval()
            for p in model.parameters():
                p.requires_grad = False
            for p in model.buffers():
                p.requires_grad = False
            return model
        _freeze_model(self.regressor)

    @staticmethod
    def sample_fmap(fmap, disp):
        """
        fmap: [B,C,H,W]
        disp: tensor of dim [BHW], disparity proposals
        return:
            sampled fmap feature of dim [B,C,H,W]
        """
        bs, _, ht, wd = fmap.shape
        device = fmap.device
        with torch.no_grad():
            grid_x = disp.reshape(bs, ht, wd)  # [B,H,W]
            grid_y = torch.zeros_like(grid_x)
            xs = torch.arange(0, wd, device=device, dtype=torch.float32).view(1, wd).expand(ht, wd)
            ys = torch.arange(0, ht, device=device, dtype=torch.float32).view(ht, 1).expand(ht, wd)
            grid = torch.stack((xs, ys), dim=-1).reshape(1, ht, wd, 2)
            grid = grid + torch.stack((-grid_x, grid_y), dim=-1)  # [B,H,W,2]
            grid[..., 0] = 2 * grid[..., 0].clone() / (wd - 1) - 1
            grid[..., 1] = 2 * grid[..., 1].clone() / (ht - 1) - 1
        feats = nn.functional.grid_sample(fmap, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return feats.reshape(bs, -1, ht, wd)
    
    @staticmethod
    def corr(fmap1, warped_fmap2):
        """
        fmap1: [B,C,H,W]
        warped_fmap2: [B,C,H,W]
        Returns:
            local cost: [B,H,W,G]
        """
        fmap1 = einops.rearrange(fmap1, 'b (g d) h w -> b g d h w', g=32)
        warped_fmap2 = einops.rearrange(warped_fmap2, 'b (g d) h w -> b g d h w', g=32)
        corr = (fmap1 * warped_fmap2).mean(dim=2)  # [B,G,H,W]
        corr = einops.rearrange(corr, 'b g h w -> b h w g')
        return corr
    
    def patch_embed(self, proposal, fmap1, fmap2, fmap1_gw, fmap2_gw, normalizer=128):
        H, W = fmap1.shape[2:]
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, proposal)  # [B,C,H,W]
        corr = self.corr(fmap1_gw, warped_fmap2_gw)  # [B,H,W,G]
        warped_fmap2 = self.sample_fmap(fmap2, proposal)  # [B,C,H,W]
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = einops.rearrange(feat_concat, 'b c h w -> b h w c')
        x = self.ffn(torch.cat((feat_concat, corr), dim=-1))
        proposal = einops.rearrange(proposal, '(b h w) -> b h w', h=H, w=W)
        pos_embed = fourier_coord_embed(proposal.unsqueeze(-1), N_freqs=15, normalizer=3.14 / normalizer)
        return x, pos_embed
    
    def forward(self, input):
        # regression network to get initial disparity
        # with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            regression_output = self.regressor(input)
        init_disp = regression_output["disp"].unsqueeze(1).float().detach()  # [B,1,H,W]
        context_raw = regression_output["feature"].float()  # [B,C,H,W]
        # context = self.feature_down(context).permute(0, 2, 3, 1).contiguous()  # [B,H/4,W/4,C]
        img1 = (input['img1'] / 255.0 - self._resnet_mean) / self._resnet_std
        img2 = (input['img2'] / 255.0 - self._resnet_mean) / self._resnet_std
        if not self.training:
            self.padder = InputPadder(img1.shape, mode="nmrf", divis_by=4)
            img1, img2, init_disp, context_raw = self.padder.pad(img1, img2, init_disp, context_raw)
        else:
            self.padder = None

        # downsample context feature to 1/4 resolution
        IH, IW = img1.shape[2:]
        context = nn.functional.interpolate(context_raw, size=(IH // 4, IW // 4), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).contiguous()
        disp_vit_init = init_disp

        img_batch = torch.cat([img1, img2], dim=0)
        fmap1, fmap2 = torch.chunk(self.extractor(img_batch), 2, dim=0)
        
        fmap1_concat = self.concatconv(fmap1)
        fmap2_concat = self.concatconv(fmap2)
        fmap1_gw = self.gw(fmap1)
        fmap2_gw = self.gw(fmap2)

        init_disp = einops.rearrange(init_disp, 'b 1 (h hs) (w ws) -> (b h w) (hs ws)', hs=4, ws=4) / 4
        init_disp = torch.median(init_disp, dim=-1, keepdim=False).values  # [Bhw]
        x, x_pos = self.patch_embed(init_disp, fmap1_concat, fmap2_concat, fmap1_gw, fmap2_gw, normalizer=128)

        # compute attention mask for SW-MSA
        attn_mask = [None]
        H, W = fmap1.shape[2:]
        if self.num_blocks > 1:
            shift_size = self.window_size // 2
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            attn_mask.append(
                WindowAttention.gen_shift_window_attn_mask((Hp, Wp), to_2tuple(self.window_size), shift_size, device=x.device)
            )

        return_intermediate = self.return_intermediate and self.training
        intermediates = []
        vit_intermediates = []
        for idx, blk in enumerate(self.blocks):
            x, context = blk(x, x_pos, attn_mask[idx % 2], context)
            if return_intermediate:
                intermediates.append(self.norm(x))
                vit_intermediates.append(context)

        if return_intermediate:
            x = torch.stack(intermediates, dim=0)
            vit = torch.stack(vit_intermediates, dim=0)
        else:
            x = self.norm(x)[None]
            vit = context[None]

        disp_update = self.head(x)  # [num_aux_layers,B,H,W,4*4]
        init_disp = einops.rearrange(init_disp, '(b h w) -> 1 b h w 1', h=H, w=W)   
        disp_all = nn.functional.relu(init_disp + disp_update)  # [num_aux_layers,B,H,W,4*4]
        disp_all = einops.rearrange(disp_all, 'n b h w (hs ws) -> n b (h hs) (w ws)', hs=4, ws=4).contiguous() * 4

        disp_vit = einops.rearrange(disp_vit_init, 'b 1 h w -> 1 b h w')
        disp_update = self.vit_refine(vit) # [num_aux_layers,B,H,W,4*4]
        disp_update = einops.rearrange(disp_update, 'n b h w (hs ws) -> n b (h hs) (w ws)', hs=4, ws=4).contiguous()
        disp_vit = nn.functional.relu(disp_vit + disp_update)

        # next stage
        # init_disp = disp_all[-1].detach()
        # init_disp = einops.rearrange(init_disp, 'b (h hs) (w ws) -> (b h w) (hs ws)', hs=4, ws=4) / 4
        # init_disp = torch.median(init_disp, dim=-1, keepdim=False).values  # [Bhw]
        # x, x_pos = self.patch_embed(init_disp, fmap1_concat, fmap2_concat, fmap1_gw, fmap2_gw, normalizer=128)

        # intermediates = []
        # vit_intermediates = []
        # for idx, blk in enumerate(self.blocks):
        #     if self.training:
        #         x, context = checkpoint(blk, x, x_pos, attn_mask[idx % 2], context, use_reentrant=self.use_reentrant)
        #     else:
        #         x, context = blk(x, x_pos, attn_mask[idx % 2], context)
        #     if return_intermediate:
        #         intermediates.append(self.norm(x))
        #         vit_intermediates.append(context)

        # if return_intermediate:
        #     x = torch.stack(intermediates, dim=0)
        #     vit = torch.stack(vit_intermediates, dim=0)
        # else:
        #     x = self.norm(x)[None]
        #     vit = context[None]

        # disp_update = self.head(x)  # [num_aux_layers,B,H,W,4*4]
        # init_disp = einops.rearrange(init_disp, '(b h w) -> 1 b h w 1', h=H, w=W)   
        # disp_all_1 = nn.functional.relu(init_disp + disp_update)  # [num_aux_layers,B,H,W,4*4]
        # disp_all_1 = einops.rearrange(disp_all_1, 'n b h w (hs ws) -> n b (h hs) (w ws)', hs=4, ws=4).contiguous() * 4

        # disp_vit_1 = einops.rearrange(disp_vit_init, 'b 1 h w -> 1 b h w')
        # disp_update = self.vit_refine(vit) # [num_aux_layers,B,H,W,4*4]
        # disp_update = einops.rearrange(disp_update, 'n b h w (hs ws) -> n b (h hs) (w ws)', hs=4, ws=4).contiguous()
        # disp_vit_1 = nn.functional.relu(disp_vit_1 + disp_update)


        disp = disp_all[-1]
        if self.padder is not None:
            disp = self.padder.unpad(disp.unsqueeze(1)).squeeze(1)

        predictions = {"disp": disp, "disp_all": disp_all, "disp_vit": disp_vit}
        # if self.training:
        #     predictions.update({"fmap1": fmap1, "fmap2": fmap2})
        return predictions