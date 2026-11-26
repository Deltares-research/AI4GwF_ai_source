from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import weakref

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ai4gwf.data import HDF5GridDataset, GridDataset, load_hdf5_files
from ai4gwf import train
from test_film import sample_arrays, write_hdf5


class LazyHDF5Tests(unittest.TestCase):
    def test_multifile_normalization_matches_eager_in_2d_and_3d(self):
        for dims in (2, 3):
            with self.subTest(dims=dims), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                arrays = sample_arrays()
                if dims == 3:
                    rng = np.random.default_rng(73)
                    arrays["grid"] = rng.random((7, 4, 6, 9), dtype=np.float32)
                    arrays["target"] = rng.random((7, 2, 4, 6, 9), dtype=np.float32)
                paths = [directory / "first.h5", directory / "second.h5"]
                write_hdf5(paths[0], {key: values[:3] for key, values in arrays.items()})
                write_hdf5(paths[1], {key: values[3:] for key, values in arrays.items()})
                specs = [[key, 0, 10 if key == "scalar" else 1] for key in arrays]
                eager = GridDataset(
                    train._normalize_arrays(load_hdf5_files(paths, arrays), specs),
                    ["grid"], "target", ["vector", "scalar"], ["forecast", "history"], dims,
                )
                lazy = HDF5GridDataset(
                    paths, [["grid", 0, 1]], ["target", 0, 1],
                    [["vector", 0, 1], ["scalar", 0, 10]],
                    [["forecast", 0, 1], ["history", 0, 1]], dims,
                )
                self.assertEqual(lazy.conditioning_layout, eager.conditioning_layout)
                for index in [0, 2, 3, 6, -1]:
                    actual_inputs, actual_target = lazy[index]
                    expected_inputs, expected_target = eager[index]
                    for key in expected_inputs:
                        torch.testing.assert_close(actual_inputs[key], expected_inputs[key], rtol=0, atol=0)
                    torch.testing.assert_close(actual_target, expected_target, rtol=0, atol=0)
                for index in [7, -8]:
                    with self.assertRaises(IndexError):
                        lazy[index]

    def test_constructor_reads_no_sample_data_even_for_large_logical_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.h5"
            count = 10_000_000
            with h5py.File(path, "w") as handle:
                handle.create_dataset("scalar", shape=(count,), dtype="f4", chunks=(1024,))
                handle.create_dataset(
                    "target", shape=(count, 1, 8, 12), dtype="f4", chunks=(1, 1, 8, 12)
                )
            with patch.object(
                h5py.Dataset, "__getitem__", side_effect=AssertionError("Sample read during construction")
            ):
                dataset = HDF5GridDataset([path], [], ["target", 0, 1], [["scalar", 0, 1]])
                self.assertEqual(len(dataset), count)
            inputs, target = dataset[count - 1]
            self.assertEqual(set(inputs), {"conditioning"})
            torch.testing.assert_close(inputs["conditioning"], torch.tensor([-1.0]))
            self.assertTrue(torch.all(target == -1))

    def test_metadata_errors_and_deferred_value_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.h5"
            arrays = sample_arrays(3)
            for changes, message in [
                ({"scalar": np.zeros(2)}, "sample count"),
                ({"scalar": np.zeros((3, 2, 2))}, "unsupported shape"),
                ({"grid": np.zeros((3, 4, 12))}, "grid dimensions"),
            ]:
                write_hdf5(path, {**arrays, **changes})
                with self.assertRaisesRegex(ValueError, message):
                    HDF5GridDataset(
                        [path], [["grid", 0, 1]], ["target", 0, 1], [["scalar", 0, 10]]
                    )
            arrays["scalar"][2] = np.nan
            write_hdf5(path, arrays)
            dataset = HDF5GridDataset([path], [], ["target", 0, 1], [["scalar", 0, 10]])
            dataset[0]
            with self.assertRaisesRegex(ValueError, "scalar.*sample 2.*non-finite"):
                dataset[2]
            with self.assertRaisesRegex(KeyError, "Missing keys"):
                HDF5GridDataset([path], [], ["target", 0, 1], [["missing", 0, 1]])
            with self.assertRaisesRegex(ValueError, "Invalid range"):
                HDF5GridDataset([path], [], ["target", 0, 1], [["scalar", 1, 1]])
            with self.assertRaisesRegex(ValueError, "distinct"):
                HDF5GridDataset([path], [["grid", 0, 1]], ["grid", 0, 1])

    def test_dataloader_workers_have_no_shared_open_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workers.h5"
            write_hdf5(path, sample_arrays(3))
            dataset = HDF5GridDataset([path], [], ["target", 0, 1], [["scalar", 0, 10]])
            restored = pickle.loads(pickle.dumps(dataset))
            batches = list(DataLoader(
                restored, batch_size=2, num_workers=2, multiprocessing_context="spawn"
            ))
            self.assertEqual(sum(len(targets) for _, targets in batches), 3)
            inputs, targets = batches[0]
            self.assertEqual(inputs["conditioning"].shape, (2, 1))
            self.assertEqual(targets.shape, (2, 2, 8, 12))

    def test_training_iterator_does_not_cache_epochs(self):
        class Loader:
            epochs = 0

            def __iter__(self):
                self.epochs += 1
                yield torch.tensor([self.epochs, 0])
                yield torch.tensor([self.epochs, 1])

        loader = Loader()
        stream = train._training_batches(loader)
        first = next(stream)
        reference = weakref.ref(first)
        del first
        next(stream)
        third = next(stream)
        self.assertEqual(third.tolist(), [2, 0])
        self.assertEqual(loader.epochs, 2)
        self.assertIsNone(reference())
        stream.close()


if __name__ == "__main__":
    unittest.main()
