from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from omegaconf import OmegaConf

from scripts.decode_vitens_barcodes import PARAMETERS, INTEGER_KEYS, decode_vitens_dataset
from scripts.preprocess_vitens_film import preprocess_vitens_dataset


class VitensDecodingTests(unittest.TestCase):
    def make_source(self, directory):
        source_path, config_path = directory / "source.h5", directory / "config.yaml"
        ranges = {"c": (-0.31, 0.31), "kd": (-0.7, 0.7), "rest": (-0.31, 244)}
        cfg = OmegaConf.create({
            "arch": {"afno": {"img_shape": [5, 19]}},
            "custom": {"data": {
                "input_keys": [[key, *bounds] for key, bounds in ranges.items()],
                "output_key": ["groundtruth", -3, 1],
            }},
        })
        OmegaConf.save(cfg, config_path)
        expected = {}
        with h5py.File(source_path, "w") as source:
            source.attrs["test_metadata"] = "preserved"
            target = source.create_dataset(
                "groundtruth", data=np.arange(3 * 5 * 19).reshape(3, 1, 5, 19)
            )
            target.attrs["units"] = "unchanged"
            for group, names in PARAMETERS.items():
                lower, upper = ranges[group]
                image = np.zeros((3, 1, 5, 19), dtype=np.float64)
                width = 19 // len(names)
                for stripe, name in enumerate(names):
                    values = (
                        np.arange(3) + 2 if name in INTEGER_KEYS
                        else np.linspace(lower + 0.1, upper - 0.1, 3)
                    )
                    expected[name] = values
                    encoded = (values - lower) / (upper - lower)
                    stop = (stripe + 1) * width if stripe < len(names) - 1 else 19
                    image[:, :, :, stripe * width:stop] = encoded[:, None, None, None]
                source[group] = image
        return source_path, config_path, expected

    def test_inverse_preserves_values_integers_remainder_and_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_path, config_path, expected = self.make_source(directory)
            output_path = directory / "decoded.h5"
            decode_vitens_dataset(source_path, config_path, output_path)
            with h5py.File(output_path, "r") as output, h5py.File(source_path, "r") as source:
                self.assertEqual(set(output), set(expected) | {"groundtruth"})
                for name, values in expected.items():
                    np.testing.assert_allclose(output[name][...], values, atol=1e-12)
                    if name in INTEGER_KEYS:
                        self.assertEqual(output[name].dtype, np.dtype("int32"))
                np.testing.assert_array_equal(output["groundtruth"], source["groundtruth"])
                self.assertEqual(output["groundtruth"].attrs["units"], "unchanged")
                self.assertEqual(output.attrs["test_metadata"], "preserved")
            with self.assertRaises(FileExistsError):
                decode_vitens_dataset(source_path, config_path, output_path)

    def test_invalid_stripes_and_noninteger_indices_create_no_output(self):
        for invalid_integer in [False, True]:
            with self.subTest(invalid_integer=invalid_integer), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source_path, config_path, _ = self.make_source(directory)
                output_path = directory / "decoded.h5"
                with h5py.File(source_path, "r+") as source:
                    if invalid_integer:
                        source["rest"][0, 0, :, 3:6] = (2.5 + 0.31) / (244 + 0.31)
                    else:
                        source["c"][0, 0, 1, 0] += 0.1
                with self.assertRaises(ValueError):
                    decode_vitens_dataset(source_path, config_path, output_path)
                self.assertFalse(output_path.exists())


class VitensPreprocessingTests(unittest.TestCase):
    def test_log_transforms_preserve_raw_values_metadata_and_sample_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source.h5"
            output_path = Path(temporary) / "film.h5"
            raw = np.array([-20000, -10000, -1000, -10, -1, 0], dtype=np.int32).reshape(3, 1, 1, 2)
            with h5py.File(source_path, "w") as source:
                source["q"] = np.array([[-1000], [-100], [-10]], dtype=np.int32)
                source["groundtruth"] = raw
                source["row"] = np.array([2, 5, 9], dtype=np.int32)
                source["x"] = [0.0, 1.0]
                source.attrs["metadata"] = "preserved"
                source["groundtruth"].attrs["units"] = "raw"
            preprocess_vitens_dataset(source_path, output_path)
            with h5py.File(output_path, "r") as output, h5py.File(source_path, "r") as source:
                for key in source:
                    np.testing.assert_array_equal(output[key][...], source[key][...])
                    self.assertEqual(output[key].dtype, source[key].dtype)
                np.testing.assert_array_equal(output["log10_negative_q"], [[3], [2], [1]])
                np.testing.assert_allclose(
                    output["groundtruth_log10"], np.log10(np.clip(raw, -10000, -1) / -1000),
                )
                self.assertEqual(output.attrs["metadata"], "preserved")
                self.assertEqual(output["groundtruth"].attrs["units"], "raw")
            with self.assertRaises(FileExistsError):
                preprocess_vitens_dataset(source_path, output_path)

    def test_invalid_q_or_target_leaves_no_output(self):
        for q, target in [(0, -1), (1, -1), (float("nan"), -1), (-1, 1), (-1, float("nan"))]:
            with self.subTest(q=q, target=target), tempfile.TemporaryDirectory() as temporary:
                source_path = Path(temporary) / "source.h5"
                output_path = Path(temporary) / "film.h5"
                with h5py.File(source_path, "w") as source:
                    source["q"] = [q]
                    source["groundtruth"] = np.full((1, 1, 2, 2), target)
                with self.assertRaises(ValueError):
                    preprocess_vitens_dataset(source_path, output_path)
                self.assertFalse(output_path.exists())
                self.assertFalse(output_path.with_suffix(".h5.partial").exists())


if __name__ == "__main__":
    unittest.main()
