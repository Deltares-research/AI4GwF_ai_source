import torch
from torch import nn
from physicsnemo.models.afno import AFNO

if __package__:
    from .afno3d import AFNO3D, reconstruct_grid
else:
    from afno3d import AFNO3D, reconstruct_grid


class FiLMAFNO(nn.Module):
    """Condition 2D/3D AFNO features, optionally starting from a learned spatial seed."""

    def __init__(
        self, backbone: AFNO | AFNO3D, conditioning_dim: int, hidden_dim: int,
        has_spatial_inputs: bool = True,
    ):
        super().__init__()
        if conditioning_dim <= 0 or hidden_dim <= 0:
            raise ValueError("FiLM conditioning_dim and hidden_dim must be positive")
        required = (
            "patch_embed", "pos_embed", "pos_drop", "blocks", "head",
            "embed_dim", "patch_size", "inp_shape", "in_chans", "out_chans",
        )
        if any(not hasattr(backbone, name) for name in required):
            raise ValueError("Unsupported PhysicsNeMo AFNO component API; use requirements.txt")
        self.backbone = backbone
        self.has_spatial_inputs = has_spatial_inputs
        self.patch_grid = tuple(
            size // patch for size, patch in zip(backbone.inp_shape, backbone.patch_size)
        )
        if not has_spatial_inputs:
            backbone.patch_embed.requires_grad_(False)
        self.conditioning_dim = conditioning_dim
        self.encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.modulations = nn.ModuleList(
            nn.Linear(hidden_dim, 2 * backbone.embed_dim) for _ in backbone.blocks
        )
        for projection in self.modulations:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self, grid: torch.Tensor | None = None, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        backbone = self.backbone
        if conditioning is None or conditioning.ndim != 2:
            raise ValueError("FiLM requires conditioning shape (B, conditioning_dim)")
        batch_size = conditioning.shape[0]
        if conditioning.shape[1] != self.conditioning_dim:
            raise ValueError(
                f"Expected conditioning shape ({batch_size}, {self.conditioning_dim}), "
                f"received {tuple(conditioning.shape)}"
            )
        if self.has_spatial_inputs:
            if grid is None or tuple(grid.shape) != (
                batch_size, backbone.in_chans, *backbone.inp_shape
            ):
                raise ValueError("Grid shape must match AFNO's configured batch, channels and image shape")
            x = backbone.patch_embed(grid) + backbone.pos_embed
        else:
            if grid is not None:
                raise ValueError("This FiLM model is configured without spatial inputs")
            x = backbone.pos_embed.expand(batch_size, -1, -1)
        encoded = self.encoder(conditioning)
        x = backbone.pos_drop(x).reshape(batch_size, *self.patch_grid, backbone.embed_dim)
        broadcast_shape = (batch_size,) + (1,) * len(self.patch_grid) + (backbone.embed_dim,)
        for block, projection in zip(backbone.blocks, self.modulations):
            x = block(x)
            gamma, beta = projection(encoded).chunk(2, dim=-1)
            x = x * (1 + gamma.reshape(broadcast_shape)) + beta.reshape(broadcast_shape)
        return reconstruct_grid(
            x, backbone.head, backbone.inp_shape, backbone.patch_size, backbone.out_chans
        )


def build_model(
    image_shape, in_channels, out_channels, config, film_config=None, conditioning_dim=0
):
    """Build PhysicsNeMo 2D or native volumetric AFNO, optionally with FiLM."""
    film_enabled = film_config.get("enabled", False) if film_config is not None else False
    if not isinstance(film_enabled, bool):
        raise ValueError("film.enabled must be a boolean")
    if in_channels < 0 or (in_channels == 0 and not film_enabled):
        raise ValueError("Spatial inputs are required without FiLM")
    if conditioning_dim > 0 and not film_enabled:
        raise ValueError("Scalar/time-series inputs require enabled FiLM")
    patch_size = config["patch_size"]
    if isinstance(patch_size, int):
        patch_size = [patch_size] * len(image_shape)
    if (
        len(image_shape) not in (2, 3) or len(patch_size) != len(image_shape)
        or any(size <= 0 for size in image_shape)
        or any(size <= 0 for size in patch_size)
        or any(size % patch for size, patch in zip(image_shape, patch_size))
    ):
        raise ValueError("AFNO requires positive 2D or 3D dimensions divisible by patch_size")
    embed_dim, num_blocks, depth = (
        int(config["embed_dim"]), int(config["num_blocks"]), int(config["depth"])
    )
    if min(embed_dim, num_blocks, depth, out_channels) <= 0:
        raise ValueError("AFNO dimensions, depth and channel counts must be positive")
    if embed_dim % num_blocks:
        raise ValueError("AFNO embed_dim must be divisible by num_blocks")
    backbone_type = AFNO if len(image_shape) == 2 else AFNO3D
    extra = (
        {"sparsity_threshold": float(config.get("sparsity_threshold", 0.01))}
        if len(image_shape) == 3 else {}
    )
    backbone = backbone_type(
        inp_shape=list(image_shape),
        in_channels=max(in_channels, 1),
        out_channels=out_channels,
        patch_size=list(patch_size),
        embed_dim=embed_dim,
        depth=depth,
        num_blocks=num_blocks,
        **extra,
    )
    if film_config is None or not film_enabled:
        return backbone
    return FiLMAFNO(
        backbone, conditioning_dim, int(film_config.get("hidden_dim", 64)),
        has_spatial_inputs=in_channels > 0,
    )
