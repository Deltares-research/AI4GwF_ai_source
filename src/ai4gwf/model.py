from physicsnemo.models.afno import AFNO


def build_model(image_shape, in_channels, out_channels, config):
    """Construct the AFNO supplied by NVIDIA PhysicsNeMo."""
    patch_size = config["patch_size"]
    if isinstance(patch_size, int):
        patch_size = [patch_size] * len(image_shape)
    return AFNO(
        inp_shape=list(image_shape),
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=list(patch_size),
        embed_dim=int(config["embed_dim"]),
        depth=int(config["depth"]),
        num_blocks=int(config["num_blocks"]),
    )
