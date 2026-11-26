import argparse
from pathlib import Path

import h5py
import numpy as np


def preprocess_vitens_dataset(source_path: Path, output_path: Path) -> None:
    """Copy raw datasets and add the original barcode parameter/target transforms."""
    source_path, output_path = Path(source_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        raise FileExistsError(partial_path)
    with h5py.File(source_path, "r") as source:
        if "log10_negative_q" in source or "groundtruth_log10" in source:
            raise ValueError("Source already contains derived FiLM datasets")
        q = np.asarray(source["q"], dtype=np.float64)
        target = source["groundtruth"]
        if q.ndim not in (1, 2) or (q.ndim == 2 and q.shape[1] != 1):
            raise ValueError("q must have shape (N,) or (N,1)")
        if not np.isfinite(q).all() or np.any(q >= 0):
            raise ValueError("log10(-q) requires finite, strictly negative q")
        if target.ndim not in (3, 4) or len(target) != len(q):
            raise ValueError("groundtruth must be a 2D grid with the same sample count as q")
        try:
            with h5py.File(partial_path, "x") as output:
                output.attrs.update(source.attrs)
                output.attrs["source_file"] = str(source_path.resolve())
                output.attrs["target_transform"] = "log10(clip(groundtruth,-10000,-1)/-1000)"
                for key, dataset in source.items():
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError("Vitens preprocessing requires flat HDF5 datasets")
                    options = (
                        {"chunks": (1, *dataset.shape[1:]), "compression": "lzf", "shuffle": True}
                        if dataset.ndim >= 3 else {}
                    )
                    copied = output.create_dataset(key, shape=dataset.shape, dtype=dataset.dtype, **options)
                    copied.attrs.update(dataset.attrs)
                    if dataset.ndim == 0:
                        copied[()] = dataset[()]
                    else:
                        for start in range(0, len(dataset), 4):
                            copied[start:start + 4] = dataset[start:start + 4]
                derived_q = output.create_dataset("log10_negative_q", data=np.log10(-q))
                derived_q.attrs["transform"] = "log10(-q)"
                transformed = output.create_dataset(
                    "groundtruth_log10", shape=target.shape, dtype="float32",
                    chunks=(1, *target.shape[1:]), compression="lzf", shuffle=True,
                )
                transformed.attrs["transform"] = output.attrs["target_transform"]
                for start in range(0, len(target), 4):
                    raw = target[start:start + 4].astype(np.float64)
                    if not np.isfinite(raw).all() or np.any(raw > 0):
                        raise ValueError(f"groundtruth contains invalid drawdown at samples {start}:{start + 4}")
                    transformed[start:start + 4] = np.log10(np.clip(raw, -10000, -1) / -1000)
                    if (start + 4) % 1000 == 0:
                        print(f"Transformed {min(start + 4, len(target))}/{len(target)} samples", flush=True)
            if output_path.exists():
                raise FileExistsError(output_path)
            partial_path.rename(output_path)
        finally:
            if partial_path.exists():
                partial_path.unlink()
    print(f"Preprocessed dataset: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare full Vitens data for scalar FiLM training")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preprocess_vitens_dataset(args.source, args.output)
