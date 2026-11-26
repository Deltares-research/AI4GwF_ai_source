from math import prod
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def reconstruct_grid(
    features: torch.Tensor, head: nn.Module, image_shape: Sequence[int],
    patch_size: Sequence[int], out_channels: int,
) -> torch.Tensor:
    """Interleave patch-grid and within-patch axes into a channels-first grid."""
    ndim = len(image_shape)
    patch_grid = [size // patch for size, patch in zip(image_shape, patch_size)]
    projected = head(features).reshape(
        features.shape[0], *patch_grid, *patch_size, out_channels
    )
    axes = [0, 2 * ndim + 1]
    for axis in range(ndim):
        axes.extend([axis + 1, axis + ndim + 1])
    return projected.permute(*axes).reshape(
        features.shape[0], out_channels, *image_shape
    )


class AFNO3DLayer(nn.Module):
    """Block-diagonal complex channel mixing over the full 3D Fourier spectrum."""

    def __init__(self, embed_dim: int, num_blocks: int, sparsity_threshold: float = 0.01):
        super().__init__()
        if embed_dim <= 0 or num_blocks <= 0 or embed_dim % num_blocks:
            raise ValueError("embed_dim must be positive and divisible by positive num_blocks")
        if not 0 <= sparsity_threshold < float("inf"):
            raise ValueError("sparsity_threshold must be finite and nonnegative")
        self.num_blocks = num_blocks
        self.block_size = embed_dim // num_blocks
        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        shape = (num_blocks, self.block_size, self.block_size, 2)
        self.w1 = nn.Parameter(0.02 * torch.randn(shape))
        self.w2 = nn.Parameter(0.02 * torch.randn(shape))
        self.b1 = nn.Parameter(0.02 * torch.randn(num_blocks, self.block_size, 2))
        self.b2 = nn.Parameter(0.02 * torch.randn(num_blocks, self.block_size, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[-1] != self.embed_dim:
            raise ValueError("AFNO3DLayer expects (B,D,H,W,embed_dim)")
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        spectrum = torch.fft.rfftn(x.to(work_dtype), dim=(1, 2, 3), norm="ortho")
        blocked = spectrum.reshape(*spectrum.shape[:-1], self.num_blocks, self.block_size)
        hidden = (
            torch.einsum(
                "...bi,bio->...bo", blocked,
                torch.view_as_complex(self.w1.to(work_dtype)),
            )
            + torch.view_as_complex(self.b1.to(work_dtype))
        )
        hidden = torch.complex(F.relu(hidden.real), F.relu(hidden.imag))
        transformed = (
            torch.einsum(
                "...bi,bio->...bo", hidden,
                torch.view_as_complex(self.w2.to(work_dtype)),
            )
            + torch.view_as_complex(self.b2.to(work_dtype))
        )
        sparse = F.softshrink(
            torch.view_as_real(transformed), lambd=self.sparsity_threshold
        )
        filtered = torch.view_as_complex(sparse.contiguous()).reshape(spectrum.shape)
        spatial = torch.fft.irfftn(
            filtered, s=x.shape[1:4], dim=(1, 2, 3), norm="ortho"
        )
        return x + spatial.to(x.dtype)


class _PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _Block3D(nn.Module):
    def __init__(self, embed_dim, num_blocks, sparsity_threshold):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.filter = AFNO3DLayer(embed_dim, num_blocks, sparsity_threshold)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.filter(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AFNO3D(nn.Module):
    """Volumetric AFNO with Conv3d patches and adaptive three-axis spectral blocks."""

    def __init__(
        self, inp_shape: Sequence[int], in_channels: int, out_channels: int,
        patch_size: Sequence[int], embed_dim: int, depth: int, num_blocks: int,
        sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        if (
            len(inp_shape) != 3 or len(patch_size) != 3
            or any(size <= 0 for size in inp_shape)
            or any(patch <= 0 for patch in patch_size)
            or any(size % patch for size, patch in zip(inp_shape, patch_size))
        ):
            raise ValueError("AFNO3D requires positive 3D dimensions divisible by patch_size")
        if min(in_channels, out_channels, embed_dim, depth, num_blocks) <= 0:
            raise ValueError("AFNO3D channels, dimensions and depth must be positive")
        self.inp_shape = list(inp_shape)
        self.patch_size = list(patch_size)
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.embed_dim = embed_dim
        self.patch_grid = tuple(size // patch for size, patch in zip(inp_shape, patch_size))
        self.patch_embed = _PatchEmbed3D(in_channels, embed_dim, self.patch_size)
        self.pos_embed = nn.Parameter(torch.empty(1, prod(self.patch_grid), embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList(
            _Block3D(embed_dim, num_blocks, sparsity_threshold) for _ in range(depth)
        )
        self.head = nn.Linear(embed_dim, out_channels * prod(patch_size), bias=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 5 or tuple(grid.shape[1:]) != (self.in_chans, *self.inp_shape):
            raise ValueError("Grid shape must match AFNO3D channels and spatial dimensions")
        x = self.pos_drop(self.patch_embed(grid) + self.pos_embed)
        x = x.reshape(grid.shape[0], *self.patch_grid, self.embed_dim)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return reconstruct_grid(
            self.forward_features(grid), self.head, self.inp_shape,
            self.patch_size, self.out_chans,
        )
