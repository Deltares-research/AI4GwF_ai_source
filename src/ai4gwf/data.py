from __future__ import annotations
from bisect import bisect_right
from operator import index as integer_index
from pathlib import Path
from typing import Dict, Iterable, Sequence
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def load_hdf5_files(paths: Sequence[str], keys: Iterable[str]) -> Dict[str, np.ndarray]:
    """Load selected arrays from one or more HDF5 files and concatenate samples."""
    keys = list(keys)
    if not paths or not keys:
        raise ValueError("At least one HDF5 file and dataset key are required")
    if len(set(keys)) != len(keys):
        raise ValueError("HDF5 dataset keys must be distinct")
    chunks = {key: [] for key in keys}
    for path in paths:
        path = str(Path(path))
        print(f"Loading {path}")
        with h5py.File(path, "r") as handle:
            missing = [key for key in keys if key not in handle]
            if missing:
                raise KeyError(f"Missing keys in {path}: {missing}")
            sample_count = None
            for key in keys:
                dataset = handle[key]
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim == 0:
                    raise ValueError(f"{path}: '{key}' must be an array with a sample axis")
                if dataset.dtype.kind not in "biuf":
                    raise ValueError(f"{path}: '{key}' must contain real numeric values")
                array = np.asarray(dataset[...], dtype=np.float32)
                _validate_array(array, f"{path}: '{key}'")
                if sample_count is not None and len(array) != sample_count:
                    raise ValueError(f"{path}: '{key}' has a mismatched sample count")
                sample_count = len(array)
                if chunks[key] and array.shape[1:] != chunks[key][0].shape[1:]:
                    raise ValueError(f"{path}: '{key}' has inconsistent shape {array.shape}")
                chunks[key].append(array)
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _validate_array(array: np.ndarray, name: str) -> None:
    if array.ndim == 0 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must have nonempty sample and feature dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def _normalization_bounds(name, lower, upper):
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper, upper - lower]).all() or upper <= lower:
        raise ValueError(f"Invalid range for '{name}': [{lower}, {upper}]")
    return lower, upper


def normalize_array(array: np.ndarray, name: str, lower: float, upper: float) -> np.ndarray:
    """Normalize into an owned float32 array without intermediate full-size copies."""
    lower, upper = _normalization_bounds(name, lower, upper)
    normalized = np.array(array, dtype=np.float32, copy=True)
    normalized -= lower
    normalized *= 2.0
    normalized /= upper - lower
    normalized -= 1.0
    _validate_array(normalized[None], name)
    return normalized


class HDF5GridDataset(Dataset):
    """Read and normalize one HDF5 sample at a time; retain only file metadata."""

    def __init__(
        self, paths: Sequence[str | Path], input_key_specs, output_key_spec,
        scalar_key_specs=(), time_series_key_specs=(), spatial_dims: int = 2,
    ):
        if not paths:
            raise ValueError("At least one HDF5 file is required")
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        self.spatial_dims = spatial_dims
        self.paths = [str(Path(path).resolve()) for path in paths]
        specs = list(input_key_specs) + list(scalar_key_specs) + list(time_series_key_specs)
        if not specs:
            raise ValueError("At least one spatial, scalar or time-series input is required")
        specs.append(output_key_spec)
        self.bounds = {}
        for spec in specs:
            if len(spec) != 3 or not isinstance(spec[0], str) or not spec[0]:
                raise ValueError(f"Invalid dataset specification: {spec}")
            name, lower, upper = spec
            if name in self.bounds:
                raise ValueError("Require distinct grid, conditioning and output keys")
            self.bounds[name] = _normalization_bounds(name, lower, upper)
        self.input_keys = [spec[0] for spec in input_key_specs]
        self.scalar_keys = [spec[0] for spec in scalar_key_specs]
        self.time_series_keys = [spec[0] for spec in time_series_key_specs]
        self.output_key = output_key_spec[0]
        self._ends = []
        shapes = {}
        total = 0
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                missing = [key for key in self.bounds if key not in handle]
                if missing:
                    raise KeyError(f"Missing keys in {path}: {missing}")
                sample_count = None
                for key in self.bounds:
                    dataset = handle[key]
                    if not isinstance(dataset, h5py.Dataset) or dataset.ndim == 0:
                        raise ValueError(f"{path}: '{key}' must have a sample axis")
                    if dataset.dtype.kind not in "biuf":
                        raise ValueError(f"{path}: '{key}' must contain real numeric values")
                    if any(size == 0 for size in dataset.shape):
                        raise ValueError(f"{path}: '{key}' must have nonempty dimensions")
                    ranks = (
                        (1, 2) if key in self.scalar_keys
                        else (2, 3) if key in self.time_series_keys
                        else (spatial_dims + 1, spatial_dims + 2)
                    )
                    if dataset.ndim not in ranks:
                        raise ValueError(f"{path}: '{key}' has unsupported shape {dataset.shape}")
                    if sample_count is not None and len(dataset) != sample_count:
                        raise ValueError(f"{path}: '{key}' has a mismatched sample count")
                    sample_count = len(dataset)
                    shape = dataset.shape[1:]
                    if key in shapes and shapes[key] != shape:
                        raise ValueError(f"{path}: '{key}' has inconsistent shape {dataset.shape}")
                    shapes[key] = shape
                total += sample_count
                self._ends.append(total)
        target_shape = shapes[self.output_key][-spatial_dims:]
        if any(shapes[key][-spatial_dims:] != target_shape for key in self.input_keys):
            raise ValueError("Spatial inputs and target must have matching grid dimensions")
        self.conditioning_layout = [
            {"key": key, "kind": kind, "shape": list(shapes[key])}
            for kind, keys in (("scalar", self.scalar_keys), ("time_series", self.time_series_keys))
            for key in keys
        ]

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("HDF5 sample index is out of range")
        file_index = bisect_right(self._ends, index)
        local_index = index - (self._ends[file_index - 1] if file_index else 0)
        path = self.paths[file_index]
        arrays = {}
        # Per-sample handles cannot leak or be inherited by DataLoader workers.
        with h5py.File(path, "r", rdcc_nbytes=1024 * 1024) as handle:
            for key, (lower, upper) in self.bounds.items():
                values = np.asarray(handle[key][local_index], dtype=np.float32)
                label = f"{path}: '{key}', sample {local_index}"
                _validate_array(values[None], label)
                arrays[key] = normalize_array(values, label, lower, upper)
        target = torch.from_numpy(
            GridDataset._ensure_channel(arrays[self.output_key][None], self.spatial_dims)[0]
        )
        grid = None
        if self.input_keys:
            grid = torch.from_numpy(np.concatenate([
                GridDataset._ensure_channel(arrays[key][None], self.spatial_dims)[0]
                for key in self.input_keys
            ], axis=0))
        keys = self.scalar_keys + self.time_series_keys
        if not keys:
            return grid, target
        inputs = {
            "conditioning": torch.from_numpy(np.concatenate([arrays[key].reshape(-1) for key in keys]))
        }
        if grid is not None:
            inputs["grid"] = grid
        return inputs, target


class GridDataset(Dataset):
    """Stack spatial inputs, with optional sample-aligned FiLM features."""
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        input_keys: Sequence[str],
        output_key: str,
        scalar_keys: Sequence[str] = (),
        time_series_keys: Sequence[str] = (),
        spatial_dims: int = 2,
    ):
        self.input_keys = list(input_keys)
        conditioning_keys = list(scalar_keys) + list(time_series_keys)
        selected_keys = self.input_keys + conditioning_keys + [output_key]
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        if not (self.input_keys or conditioning_keys):
            raise ValueError("At least one spatial, scalar or time-series input is required")
        if len(set(selected_keys)) != len(selected_keys):
            raise ValueError("Require distinct grid, conditioning and output keys")
        for key in selected_keys:
            _validate_array(arrays[key], f"'{key}'")
            if len(arrays[key]) != len(arrays[output_key]):
                raise ValueError(f"'{key}' and '{output_key}' have different sample counts")
        x_arrays = [self._ensure_channel(arrays[key], spatial_dims) for key in self.input_keys]
        target = self._ensure_channel(arrays[output_key], spatial_dims)
        if any(array.shape[2:] != target.shape[2:] for array in x_arrays):
            raise ValueError("Spatial inputs and target must have matching grid dimensions")
        self.x = torch.from_numpy(np.concatenate(x_arrays, axis=1)).float() if x_arrays else None
        self.y = torch.from_numpy(target).float()
        self.conditioning = None
        self.conditioning_layout = []
        features = []
        for kind, keys, ranks in (
            ("scalar", scalar_keys, (1, 2)),
            ("time_series", time_series_keys, (2, 3)),
        ):
            for key in keys:
                array = arrays[key]
                if array.ndim not in ranks:
                    raise ValueError(f"'{key}' ({kind}) has unsupported shape {array.shape}")
                features.append(array.reshape(len(array), -1))
                self.conditioning_layout.append(
                    {"key": key, "kind": kind, "shape": list(array.shape[1:])}
                )
        if features:
            self.conditioning = torch.from_numpy(np.concatenate(features, axis=1)).float()

    @staticmethod
    def _ensure_channel(array: np.ndarray, spatial_dims: int = 2) -> np.ndarray:
        if array.ndim == spatial_dims + 1:
            return array[:, None, ...]
        if array.ndim == spatial_dims + 2:
            return array
        raise ValueError(f"Expected {spatial_dims} spatial axes with optional channels, received {array.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.conditioning is None:
            return self.x[index], self.y[index]
        inputs = {"conditioning": self.conditioning[index]}
        if self.x is not None:
            inputs["grid"] = self.x[index]
        return inputs, self.y[index]


def make_train_validation_datasets(
    arrays, input_keys, output_key, train_fraction, seed, scalar_keys=(), time_series_keys=(),
    spatial_dims=2,
):
    dataset = GridDataset(
        arrays, input_keys, output_key, scalar_keys, time_series_keys, spatial_dims
    )
    return split_train_validation_dataset(dataset, train_fraction, seed)


def split_train_validation_dataset(dataset: Dataset, train_fraction: float, seed: int):
    if len(dataset) < 2:
        raise ValueError("At least two samples are required for training and validation")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    n_train = int(round(len(dataset) * train_fraction))
    n_train = min(max(n_train, 1), len(dataset) - 1)
    return random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(seed),
    )
