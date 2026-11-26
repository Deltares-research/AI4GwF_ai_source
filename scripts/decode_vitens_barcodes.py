"""Recover Vitens's 20 pre-barcode parameters without changing spatial targets."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf


PARAMETERS = {
    "c": [f"log10_c{layer}_fct" for layer in range(1, 8)],
    "kd": [f"log10_kd{layer}_fct" for layer in range(1, 9)],
    "rest": ["log10_rch_fct", "row", "col", "layer", "log10_negative_q"],
}
INTEGER_KEYS = {"row", "col", "layer"}


def decode_vitens_dataset(source_path: Path, config_path: Path, output_path: Path) -> None:
    """Invert constant vertical stripes and their configured linear normalization."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    cfg = OmegaConf.load(config_path)
    ranges = {}
    for key, lower, upper in cfg.custom.data.input_keys:
        if key in ranges:
            raise ValueError(f"Duplicate barcode key: {key}")
        lower, upper = float(lower), float(upper)
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise ValueError(f"Invalid barcode bounds for '{key}'")
        ranges[key] = (lower, upper)
    if set(ranges) != set(PARAMETERS):
        raise ValueError("Vitens conversion requires exactly c, kd and rest barcode keys")
    output_specs = cfg.custom.data.output_key
    if isinstance(output_specs[0], str):
        output_specs = [output_specs]
    if len(output_specs) != 1:
        raise ValueError("Vitens conversion requires exactly one target dataset")
    target_key = str(output_specs[0][0])
    image_shape = tuple(cfg.arch.afno.img_shape)
    if len(image_shape) != 2:
        raise ValueError("Vitens barcodes must be 2D")

    with h5py.File(source_path, "r") as source:
        target = source[target_key]
        if target.ndim != 4 or tuple(target.shape[2:]) != image_shape or len(target) == 0:
            raise ValueError("Target must have shape (N,C,H,W) matching the config")
        sample_count = len(target)
        for index in range(len(target)):
            if not np.isfinite(target[index]).all():
                raise ValueError(f"Target sample {index} contains non-finite values")
        recovered = {}
        for group, names in PARAMETERS.items():
            barcode = source[group]
            if tuple(barcode.shape) != (len(target), 1, *image_shape):
                raise ValueError(f"'{group}' must have shape (N,1,H,W) matching the target")
            width = image_shape[1] // len(names)
            if width == 0:
                raise ValueError(f"'{group}' barcode is too narrow for {len(names)} stripes")
            values = np.empty((len(target), len(names)), dtype=np.float64)
            lower, upper = ranges[group]
            for index in range(len(target)):
                image = barcode[index, 0]
                for stripe, name in enumerate(names):
                    start = stripe * width
                    stop = (stripe + 1) * width if stripe < len(names) - 1 else image_shape[1]
                    value = image[0, start]
                    if not np.isfinite(value) or not np.all(image[:, start:stop] == value):
                        raise ValueError(f"'{group}', sample {index}, stripe {stripe} is not constant")
                    values[index, stripe] = value * (upper - lower) + lower
            if not np.isfinite(values).all():
                raise ValueError(f"'{group}' decoding produced non-finite values")
            for stripe, name in enumerate(names):
                column = values[:, stripe]
                if name in INTEGER_KEYS:
                    rounded = np.rint(column)
                    if (
                        np.any(np.abs(column - rounded) > 1e-8)
                        or np.any(rounded < np.iinfo(np.int32).min)
                        or np.any(rounded > np.iinfo(np.int32).max)
                    ):
                        raise ValueError(f"'{name}' does not decode to int32 values")
                    column = rounded.astype(np.int32)
                recovered[name] = column

        with h5py.File(output_path, "x") as output:
            for name, value in source.attrs.items():
                output.attrs[name] = value
            output.attrs["source_barcode_file"] = str(source_path.resolve())
            output.attrs["source_config_file"] = str(config_path.resolve())
            output.attrs["parameter_representation"] = "pre_barcode_log10"
            output.attrs["parameter_order"] = list(recovered)
            source.copy(target_key, output)
            for group, names in PARAMETERS.items():
                for stripe, name in enumerate(names):
                    dataset = output.create_dataset(name, data=recovered[name])
                    dataset.attrs["source_barcode_key"] = group
                    dataset.attrs["stripe_index"] = stripe
                    dataset.attrs["barcode_lower"], dataset.attrs["barcode_upper"] = ranges[group]
                    dataset.attrs["representation"] = "integer" if name in INTEGER_KEYS else "log10"
    print(f"Created {output_path}: {sample_count} samples, 20 scalar/integer inputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decode_vitens_dataset(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
