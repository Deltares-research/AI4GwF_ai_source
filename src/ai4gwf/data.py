from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import os

def load_hdf5_files(paths: Sequence[str], keys: Iterable[str]) -> Dict[str, np.ndarray]:
    """Load selected arrays from one or more HDF5 files and concatenate samples."""
    keys = list(keys)
    chunks = {key: [] for key in keys}
    for path in paths:
        path = str(Path(path))
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Loading {path}")
        with h5py.File(path, "r") as handle:
            missing = [key for key in keys if key not in handle]
            if missing:
                raise KeyError(f"Missing keys in {path}: {missing}")
            for key in keys:
                chunks[key].append(np.asarray(handle[key][:], dtype=np.float32))
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


class GridDataset(Dataset):
    """Return AFNO inputs as one channel-stacked tensor and one target tensor."""
    def __init__(self, arrays: Dict[str, np.ndarray], input_keys: Sequence[str], output_key: str):
        self.input_keys = list(input_keys)
        x_arrays = [self._ensure_channel(arrays[key]) for key in self.input_keys]
        self.x = torch.from_numpy(np.concatenate(x_arrays, axis=1)).float()
        self.y = torch.from_numpy(self._ensure_channel(arrays[output_key])).float()
        if len(self.x) != len(self.y):
            raise ValueError("Inputs and targets have different sample counts")

    @staticmethod
    def _ensure_channel(array: np.ndarray) -> np.ndarray:
        if array.ndim == 3:  # N,H,W -> N,1,H,W
            return array[:, None, :, :]
        if array.ndim == 4:  # N,C,H,W
            return array
        raise ValueError(f"Expected N,H,W or N,C,H,W, received {array.shape}")

    def __len__(self): return len(self.x)
    def __getitem__(self, index): return self.x[index], self.y[index]


def make_train_validation_datasets(arrays, input_keys, output_key, train_fraction, seed):
    dataset = GridDataset(arrays, input_keys, output_key)
    n_train = int(round(len(dataset) * train_fraction))
    n_train = min(max(n_train, 1), len(dataset) - 1)
    return random_split(dataset, [n_train, len(dataset)-n_train], generator=torch.Generator().manual_seed(seed))



