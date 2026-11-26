import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from physicsnemo.models.afno import AFNO

from ai4gwf.data import GridDataset, load_hdf5_files, make_train_validation_datasets
from ai4gwf.model import FiLMAFNO, build_model
from ai4gwf.afno3d import AFNO3D, AFNO3DLayer, reconstruct_grid
from ai4gwf import train


ROOT = Path(__file__).resolve().parents[1]
ARCH = {
    "img_shape": [8, 12], "patch_size": [2, 3],
    "embed_dim": 8, "depth": 2, "num_blocks": 2,
}


def sample_arrays(n=7):
    rng = np.random.default_rng(42)
    return {
        "grid": rng.random((n, 8, 12), dtype=np.float32),
        "target": rng.random((n, 2, 8, 12), dtype=np.float32),
        "scalar": np.arange(n, dtype=np.float32),
        "vector": rng.random((n, 2), dtype=np.float32),
        "history": rng.random((n, 3), dtype=np.float32),
        "forecast": rng.random((n, 2, 2), dtype=np.float32),
    }


def write_hdf5(path, arrays):
    with h5py.File(path, "w") as handle:
        for key, array in arrays.items():
            handle[key] = array


class MixedDataTests(unittest.TestCase):
    def test_multifile_loading_normalization_and_feature_order(self):
        arrays = sample_arrays()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{index}.h5" for index in range(2)]
            write_hdf5(paths[0], {key: value[:3] for key, value in arrays.items()})
            write_hdf5(paths[1], {key: value[3:] for key, value in arrays.items()})
            loaded = load_hdf5_files(paths, arrays)
        for key in arrays:
            np.testing.assert_array_equal(loaded[key], arrays[key])
        normalized = train._normalize_arrays(loaded, [["scalar", 0, 6]])
        np.testing.assert_allclose(normalized["scalar"][[0, -1]], [-1, 1])
        dataset = GridDataset(
            normalized, ["grid"], "target", ["vector", "scalar"], ["forecast", "history"]
        )
        inputs, target = dataset[2]
        self.assertEqual(inputs["grid"].shape, (1, 8, 12))
        self.assertEqual(target.shape, (2, 8, 12))
        expected = np.concatenate([
            normalized[key][2].reshape(-1)
            for key in ["vector", "scalar", "forecast", "history"]
        ])
        np.testing.assert_array_equal(inputs["conditioning"].numpy(), expected)
        self.assertEqual(
            dataset.conditioning_layout,
            [
                {"key": "vector", "kind": "scalar", "shape": [2]},
                {"key": "scalar", "kind": "scalar", "shape": []},
                {"key": "forecast", "kind": "time_series", "shape": [2, 2]},
                {"key": "history", "kind": "time_series", "shape": [3]},
            ],
        )
        plain = GridDataset(loaded, ["grid"], "target")
        self.assertIsInstance(plain[0][0], torch.Tensor)
        stacked = GridDataset(
            {**loaded, "grid2": loaded["target"]}, ["grid", "grid2"], "target"
        )
        self.assertEqual(stacked.x.shape, (7, 3, 8, 12))
        first = make_train_validation_datasets(
            loaded, ["grid"], "target", 0.6, 42, ["scalar"], ["history"]
        )
        second = make_train_validation_datasets(
            loaded, ["grid"], "target", 0.6, 42, ["scalar"], ["history"]
        )
        self.assertEqual(first[0].indices, second[0].indices)
        self.assertEqual(len(first[0]) + len(first[1]), 7)

    def test_loader_rejects_malformed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.h5", Path(directory) / "second.h5"
            cases = [
                ({"grid": np.zeros((2, 8, 12))}, KeyError, "Missing keys"),
                ({"grid": np.zeros((2, 8, 12)), "target": np.zeros((3, 8, 12))},
                 ValueError, "sample count"),
                ({"grid": np.full((2, 8, 12), np.nan), "target": np.zeros((2, 8, 12))},
                 ValueError, "non-finite"),
                ({"grid": np.zeros((0, 8, 12)), "target": np.zeros((0, 8, 12))},
                 ValueError, "nonempty"),
                ({"grid": np.array(1.0), "target": np.zeros((2, 8, 12))},
                 ValueError, "sample axis"),
                ({"grid": np.array([b"bad"]), "target": np.zeros((1, 8, 12))},
                 ValueError, "numeric"),
            ]
            for arrays, error, message in cases:
                with self.subTest(message=message):
                    write_hdf5(first, arrays)
                    with self.assertRaisesRegex(error, message):
                        load_hdf5_files([first], ["grid", "target"])
            write_hdf5(first, {"grid": np.zeros((2, 8, 12))})
            write_hdf5(second, {"grid": np.zeros((2, 9, 12))})
            with self.assertRaisesRegex(ValueError, "inconsistent shape"):
                load_hdf5_files([first, second], ["grid"])

    def test_dataset_rejects_bad_shapes_and_splits(self):
        arrays = sample_arrays()
        cases = [
            ({"scalar": np.zeros((7, 1, 1))}, ["scalar"], [], "unsupported shape"),
            ({"history": np.zeros(7)}, [], ["history"], "unsupported shape"),
            ({"scalar": np.zeros(6)}, ["scalar"], [], "sample counts"),
            ({"grid": np.zeros((7, 4, 12))}, [], [], "grid dimensions"),
            ({"grid": np.zeros((7, 2))}, [], [], "Expected"),
            ({"scalar": np.full(7, np.inf)}, ["scalar"], [], "non-finite"),
            ({}, ["grid"], [], "distinct"),
            ({"history": np.zeros((7, 0))}, [], ["history"], "nonempty"),
        ]
        for changes, scalars, sequences, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    GridDataset({**arrays, **changes}, ["grid"], "target", scalars, sequences)
        with self.assertRaisesRegex(ValueError, "two samples"):
            make_train_validation_datasets(sample_arrays(1), ["grid"], "target", 0.8, 42)
        for fraction in [0, 1, float("nan")]:
            with self.assertRaisesRegex(ValueError, "train_fraction"):
                make_train_validation_datasets(arrays, ["grid"], "target", fraction, 42)
        for bounds in [(1, 1), (2, 1), (0, float("inf")), (float("nan"), 1)]:
            with self.assertRaisesRegex(ValueError, "Invalid range"):
                train._normalize_arrays(arrays, [["scalar", *bounds]])
        for specs in ["grid", [["grid", 0]], [[None, 0, 1]]]:
            with self.assertRaises(ValueError):
                train._normalize_key_specs(specs)


class FiLMModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def build(self, enabled=True):
        return build_model(
            [8, 12], 1, 2, ARCH, {"enabled": enabled, "hidden_dim": 16}, 4 if enabled else 0
        )

    def test_disabled_factory_preserves_plain_afno(self):
        model = self.build(False)
        self.assertIs(type(model), AFNO)
        original = AFNO(
            inp_shape=[8, 12], in_channels=1, out_channels=2,
            patch_size=[2, 3], embed_dim=8, depth=2, num_blocks=2,
        )
        self.assertEqual(set(model.state_dict()), set(original.state_dict()))
        original.load_state_dict(model.state_dict())
        grid = torch.randn(2, 1, 8, 12)
        torch.testing.assert_close(original(grid), model(grid))
        self.assertIs(type(build_model([8, 12], 1, 2, ARCH)), AFNO)

    def test_identity_gradients_conditioning_effect_and_roundtrip(self):
        model = self.build().eval()
        self.assertIsInstance(model, FiLMAFNO)
        grid = torch.randn(2, 1, 8, 12)
        conditioning = torch.randn(2, 4)
        torch.testing.assert_close(
            model(grid, conditioning), model.backbone(grid), rtol=0, atol=0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        target = torch.randn(2, 2, 8, 12)
        ((model(grid, conditioning) - target) ** 2).mean().backward()
        self.assertGreater(model.modulations[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.head.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        ((model(grid, conditioning) - target) ** 2).mean().backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        output = model(grid, conditioning)
        self.assertEqual(output.shape, (2, 2, 8, 12))
        self.assertGreater(
            (output - model(grid, conditioning + 2)).abs().max().item(), 1e-6
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = self.build().eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(restored(grid, conditioning), output)

    def test_model_rejects_invalid_configuration_and_inputs(self):
        for enabled in ["false", 1]:
            with self.assertRaisesRegex(ValueError, "boolean"):
                build_model([8, 12], 1, 2, ARCH, {"enabled": enabled}, 4)
        for change in [
            {"patch_size": 5}, {"embed_dim": 7}, {"depth": 0}, {"num_blocks": 0}
        ]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    build_model([8, 12], 1, 2, {**ARCH, **change})
        for width, hidden in [(0, 16), (4, 0)]:
            with self.assertRaisesRegex(ValueError, "positive"):
                build_model(
                    [8, 12], 1, 2, ARCH,
                    {"enabled": True, "hidden_dim": hidden}, width,
                )
        model = self.build()
        with self.assertRaisesRegex(ValueError, "conditioning shape"):
            model(torch.zeros(2, 1, 8, 12), torch.zeros(2, 5))
        with self.assertRaisesRegex(ValueError, "Grid shape"):
            model(torch.zeros(2, 1, 12, 8), torch.zeros(2, 4))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_evaluation_weights_partial_batches_by_sample_count(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, grid, conditioning):
                return torch.zeros_like(grid)

        dataset = GridDataset(
            {
                "grid": np.zeros((3, 2, 2), dtype=np.float32),
                "target": np.broadcast_to(
                    np.arange(1, 4, dtype=np.float32)[:, None, None], (3, 2, 2)
                ).copy(),
                "scalar": np.zeros(3, dtype=np.float32),
            },
            ["grid"], "target", ["scalar"],
        )
        loss = train._evaluate(
            ZeroModel(), DataLoader(dataset, batch_size=2), torch.device("cpu")
        )
        self.assertAlmostEqual(loss, 14 / 3)

    def test_enabled_without_conditioning_fails_before_loading(self):
        cfg = OmegaConf.load(ROOT / "configs/config_afno.yaml")
        cfg.film.enabled = True
        with patch.object(
            train, "parse_args",
            return_value=SimpleNamespace(config_dir="/tmp", config_name="bad"),
        ), patch.object(train.OmegaConf, "load", return_value=cfg):
            with self.assertRaisesRegex(ValueError, "requires scalar_keys"):
                train.main()

    def test_epoch_snapshots_are_retained_when_validation_worsens(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cfg = OmegaConf.load(ROOT / "configs/config_afno_film.yaml")
            cfg.arch.afno = ARCH
            cfg.batch_size = {"grid": 2, "validation": 2}
            cfg.training = {"max_steps": 4, "rec_validation_freq": 99}
            cfg.custom.data.input_keys = [["grid", 0, 1]]
            cfg.custom.data.output_key = ["target", 0, 1]
            cfg.custom.data.scalar_keys = [["scalar", 0, 10]]
            cfg.custom.data.time_series_keys = []
            cfg.custom.data.train_fraction = 0.7
            data_path = directory / "data.h5"
            write_hdf5(data_path, sample_arrays(6))
            cfg.custom.data.files = [str(data_path)]
            cfg.custom.output.directory = str(directory / "output")
            cfg.custom.output.visualize_each_epoch = True
            cfg.custom.output.examples_to_plot = 1
            with patch.object(train, "parse_args", return_value=SimpleNamespace(
                config_dir=str(directory), config_name="test",
            )), patch.object(train.OmegaConf, "load", return_value=cfg), patch.object(
                train, "_evaluate", side_effect=[0.1, 0.2],
            ), patch.object(torch.cuda, "is_available", return_value=False):
                train.main()
            output = Path(cfg.custom.output.directory)
            for epoch in (1, 2):
                self.assertTrue((output / "epochs" / f"epoch_{epoch:04d}" / "predictions.npz").is_file())
            history = json.loads((output / "epochs" / "loss_history.json").read_text())
            self.assertEqual([record["validation_mse"] for record in history], [0.1, 0.2])
            checkpoint = torch.load(output / cfg.custom.output.checkpoint_name, weights_only=True)
            self.assertEqual(checkpoint["step"], 2)

    def test_real_afno_training_both_entry_points_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            arrays = sample_arrays(9)
            for enabled, entry_point in [
                (False, [str(ROOT / "src/ai4gwf/train.py")]),
                (True, ["-m", "ai4gwf.train"]),
                (None, ["-m", "ai4gwf.train"]),
            ]:
                with self.subTest(film_enabled=enabled):
                    cfg = OmegaConf.load(
                        ROOT / "configs" /
                        ("config_afno_film.yaml" if enabled else "config_afno.yaml")
                    )
                    cfg.arch.afno = ARCH
                    if enabled is None:
                        del cfg.film
                    cfg.batch_size = {"grid": 4, "validation": 2}
                    cfg.training = {"max_steps": 2, "rec_validation_freq": 2, "log_freq": 1}
                    cfg.custom.data.input_keys = [["grid", 0, 1]]
                    cfg.custom.data.output_key = ["target", 0, 1]
                    cfg.custom.data.train_fraction = 0.7
                    cfg.custom.data.scalar_keys = (
                        [["scalar", 0, 10], ["vector", 0, 1]] if enabled else []
                    )
                    cfg.custom.data.time_series_keys = [
                        ["history", 0, 1], ["forecast", 0, 1]
                    ] if enabled else []
                    output_dir = directory / str(enabled)
                    cfg.custom.output.directory = str(output_dir)
                    cfg.custom.output.examples_to_plot = 2
                    cfg.custom.output.visualize_each_epoch = bool(enabled)
                    data_path = directory / f"{enabled}.h5"
                    write_hdf5(
                        data_path, arrays if enabled else {
                            key: arrays[key] for key in ["grid", "target"]
                        }
                    )
                    cfg.custom.data.files = [os.path.relpath(data_path, ROOT)]
                    name = f"config_{enabled}"
                    OmegaConf.save(cfg, directory / f"{name}.yaml")
                    env = {
                        **os.environ, "PYTHONPATH": str(ROOT / "src"),
                        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    }
                    result = subprocess.run(
                        [sys.executable, *entry_point,
                         "--config_dir", os.path.relpath(directory, ROOT),
                         "--config_name", name],
                        cwd=directory, env=env, capture_output=True, text=True, timeout=120,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("step=1 train_mse=", result.stdout)
                    self.assertIn("step=2", result.stdout)
                    if enabled:
                        epoch_dir = output_dir / "epochs" / "epoch_0001"
                        self.assertTrue((epoch_dir / "predictions.npz").is_file())
                        for channel in range(2):
                            self.assertGreater(
                                (epoch_dir / f"comparison_channel_{channel:02d}.png").stat().st_size,
                                1000,
                            )
                        history = json.loads((output_dir / "epochs" / "loss_history.json").read_text())
                        self.assertEqual(history[0]["step"], 2)
                        self.assertTrue(np.isfinite(history[0]["train_mse"]))
                        self.assertTrue((output_dir / "epochs" / "loss_curve.png").is_file())
                    else:
                        self.assertFalse((output_dir / "epochs").exists())
                    checkpoint = torch.load(
                        output_dir / "best.pt", map_location="cpu", weights_only=True
                    )
                    self.assertTrue(np.isfinite(checkpoint["validation_mse"]))
                    with np.load(output_dir / "predictions.npz") as exported:
                        self.assertEqual(exported["inputs"].shape, (2, 1, 8, 12))
                        self.assertEqual(exported["predictions"].shape, (2, 2, 8, 12))
                        self.assertTrue(np.isfinite(exported["predictions"]).all())
                        if enabled:
                            self.assertEqual(checkpoint["conditioning_dim"], 10)
                            self.assertEqual(exported["conditioning"].shape, (2, 10))
                            self.assertEqual(
                                json.loads(str(exported["conditioning_layout"])),
                                checkpoint["conditioning_layout"],
                            )
                            restored = build_model(
                                [8, 12], 1, 2, checkpoint["config"]["arch"]["afno"],
                                checkpoint["config"]["film"], checkpoint["conditioning_dim"],
                            ).eval()
                            restored.load_state_dict(checkpoint["model_state_dict"])
                            with torch.no_grad():
                                predicted = restored(
                                    torch.from_numpy(exported["inputs"]),
                                    torch.from_numpy(exported["conditioning"]),
                                ).numpy()
                            np.testing.assert_allclose(
                                (predicted + 1) / 2, exported["predictions"],
                                rtol=1e-5, atol=1e-6,
                            )
                        else:
                            self.assertEqual(
                                set(exported.files), {"inputs", "targets", "predictions"}
                            )
                            self.assertNotIn("conditioning_dim", checkpoint)


class VolumetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_epoch_visualizations_use_central_depth_slice(self):
        from ai4gwf.plotting import save_epoch_plots

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "epochs" / "epoch_0001"
            directory.mkdir(parents=True)
            targets = np.zeros((1, 1, 3, 4, 5), dtype=np.float32)
            targets[:, :, 1] = 0.5
            path = directory / "predictions.npz"
            np.savez(path, targets=targets, predictions=targets + 0.1)
            history = [{"epoch": 1, "step": 2, "train_mse": 0.1, "validation_mse": 0.2}]
            save_epoch_plots(path, history, ["target", 0, 1], directory)
            self.assertGreater((directory / "comparison_channel_00.png").stat().st_size, 1000)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["targets"], targets)

    def test_patch_reconstruction_preserves_volume_coordinates_and_channels(self):
        volume = torch.arange(2 * 4 * 6 * 9).reshape(1, 2, 4, 6, 9)
        patches = volume.reshape(1, 2, 2, 2, 3, 2, 3, 3)
        features = patches.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, 2, 3, 3, 24)
        restored = reconstruct_grid(
            features, torch.nn.Identity(), [4, 6, 9], [2, 2, 3], 2
        )
        torch.testing.assert_close(restored, volume)

    def test_spectral_filter_math_and_nonlocal_coupling_on_all_three_axes(self):
        layer = AFNO3DLayer(4, 2, sparsity_threshold=0).double()
        with torch.no_grad():
            for weight in [layer.w1, layer.w2]:
                weight.zero_()
                weight[..., 0].copy_(torch.eye(2).expand(2, 2, 2))
            layer.b1.zero_()
            layer.b2.zero_()
        shape = (5, 6, 7)
        signal = torch.zeros(1, *shape, 4, dtype=torch.float64)
        for axis, size in enumerate(shape, start=1):
            coordinate = torch.arange(size, dtype=torch.float64)
            wave = torch.cos(2 * torch.pi * coordinate / size)
            broadcast = [1] * 5
            broadcast[axis] = size
            signal += wave.reshape(broadcast)
        torch.testing.assert_close(layer(signal), 2 * signal, rtol=1e-10, atol=1e-10)
        for axis in (1, 2, 3):
            with self.subTest(axis=axis):
                slab = torch.zeros_like(signal)
                slab.select(axis, 1).fill_(1)
                response = layer(slab) - slab
                self.assertGreater(response.select(axis, 0).abs().max().item(), 1e-3)

    def test_spectral_gradients_and_sparsity(self):
        torch.manual_seed(12)
        layer = AFNO3DLayer(2, 1, sparsity_threshold=0).double()
        x = torch.randn(1, 2, 3, 3, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(layer, (x,), fast_mode=True))
        layer(x).square().mean().backward()
        self.assertGreater(layer.w1.grad.abs().sum().item(), 0)
        self.assertGreater(layer.w2.grad.abs().sum().item(), 0)
        layer.sparsity_threshold = 1e6
        torch.testing.assert_close(layer(x), x, rtol=0, atol=0)
        for threshold in [-1, float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "sparsity_threshold"):
                AFNO3DLayer(4, 2, threshold)


class AllInputCombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_conditioning_only_seed_learns_in_2d_and_3d(self):
        for shape, patches in [([8, 12], [2, 3]), ([4, 6, 9], [2, 2, 3])]:
            with self.subTest(shape=shape):
                torch.manual_seed(23)
                arch = {**ARCH, "patch_size": patches}
                model = build_model(
                    shape, 0, 2, arch, {"enabled": True, "hidden_dim": 16}, 4
                ).eval()
                conditioning = torch.randn(2, 4)
                initial = model(conditioning=conditioning)
                torch.testing.assert_close(
                    initial, model(conditioning=conditioning + 2), rtol=0, atol=0
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                target = torch.randn(2, 2, *shape)
                for _ in range(2):
                    optimizer.zero_grad()
                    ((model(conditioning=conditioning) - target) ** 2).mean().backward()
                    optimizer.step()
                self.assertGreater(model.backbone.pos_embed.grad.abs().sum().item(), 0)
                self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(
                    not parameter.requires_grad
                    for parameter in model.backbone.patch_embed.parameters()
                ))
                prediction = model(conditioning=conditioning)
                self.assertGreater(
                    (prediction - model(conditioning=conditioning + 2)).abs().max().item(),
                    1e-6,
                )
                with self.assertRaisesRegex(ValueError, "without spatial inputs"):
                    model(torch.zeros(2, 1, *shape), conditioning)
                with self.assertRaisesRegex(ValueError, "conditioning shape"):
                    model()
                spatial = build_model(
                    shape, 1, 2, arch, {"enabled": True, "hidden_dim": 16}, 4
                ).eval()
                grid = torch.randn(2, 1, *shape)
                torch.testing.assert_close(
                    spatial(grid, conditioning), spatial.backbone(grid), rtol=0, atol=0
                )

    def test_rejects_empty_inputs_disabled_conditioning_and_spatial_mismatches(self):
        arrays = sample_arrays()
        with self.assertRaisesRegex(ValueError, "At least one"):
            GridDataset(arrays, [], "target")
        with self.assertRaisesRegex(ValueError, "require enabled FiLM"):
            build_model([8, 12], 1, 2, ARCH, {"enabled": False}, 2)
        with self.assertRaisesRegex(ValueError, "Spatial inputs are required"):
            build_model([8, 12], 0, 2, ARCH)
        volume = np.zeros((7, 4, 6, 9), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "spatial axes"):
            GridDataset({**arrays, "target": volume}, ["grid"], "target", spatial_dims=3)
        with self.assertRaisesRegex(ValueError, "grid dimensions"):
            GridDataset(
                {"grid": volume, "target": np.zeros((7, 5, 6, 9), dtype=np.float32)},
                ["grid"], "target", spatial_dims=3,
            )
        cfg = OmegaConf.load(ROOT / "configs/config_afno.yaml")
        cfg.custom.data.scalar_keys = [["scalar", 0, 1]]
        with patch.object(
            train, "parse_args",
            return_value=SimpleNamespace(config_dir="/tmp", config_name="bad"),
        ), patch.object(train.OmegaConf, "load", return_value=cfg):
            with self.assertRaisesRegex(ValueError, "require enabled FiLM"):
                train.main()

    def test_training_exports_and_checkpoints_for_every_input_combination(self):
        rng = np.random.default_rng(73)
        combinations = [
            flags for flags in product([False, True], repeat=3) if any(flags)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for dims, (grids, scalars, sequences) in product([2, 3], combinations):
                with self.subTest(dims=dims, grids=grids, scalars=scalars, sequences=sequences):
                    shape = [8, 12] if dims == 2 else [4, 6, 9]
                    patches = [2, 3] if dims == 2 else [2, 2, 3]
                    name = f"{dims}_{int(grids)}{int(scalars)}{int(sequences)}"
                    arrays = {"target": rng.random((9, 2, *shape), dtype=np.float32)}
                    if grids:
                        arrays["grid"] = rng.random((9, *shape), dtype=np.float32)
                        arrays["grid2"] = rng.random((9, 2, *shape), dtype=np.float32)
                    if scalars:
                        arrays["scalar"] = rng.random(9, dtype=np.float32)
                        arrays["vector"] = rng.random((9, 2), dtype=np.float32)
                    if sequences:
                        arrays["history"] = rng.random((9, 3), dtype=np.float32)
                        arrays["forecast"] = rng.random((9, 2, 2), dtype=np.float32)
                    data_path = directory / f"{name}.h5"
                    write_hdf5(data_path, arrays)
                    cfg = OmegaConf.load(ROOT / "configs/config_afno.yaml")
                    cfg.arch.afno = {**ARCH, "img_shape": shape, "patch_size": patches}
                    enabled = scalars or sequences
                    cfg.film = {"enabled": enabled, "hidden_dim": 16}
                    cfg.batch_size = {"grid": 4, "validation": 2}
                    cfg.training = {"max_steps": 2, "rec_validation_freq": 1}
                    cfg.custom.data.files = [str(data_path)]
                    cfg.custom.data.input_keys = (
                        [["grid", 0, 1], ["grid2", 0, 1]] if grids else []
                    )
                    cfg.custom.data.scalar_keys = (
                        [["scalar", 0, 1], ["vector", 0, 1]] if scalars else []
                    )
                    cfg.custom.data.time_series_keys = (
                        [["history", 0, 1], ["forecast", 0, 1]] if sequences else []
                    )
                    cfg.custom.data.output_key = ["target", 0, 1]
                    cfg.custom.data.train_fraction = 0.7
                    cfg.custom.output.directory = str(directory / name)
                    cfg.custom.output.examples_to_plot = 2
                    OmegaConf.save(cfg, directory / f"{name}.yaml")
                    with patch.object(
                        train, "parse_args",
                        return_value=SimpleNamespace(config_dir=str(directory), config_name=name),
                    ), patch.object(torch.cuda, "is_available", return_value=False):
                        train.main()
                    output_dir = directory / name
                    checkpoint = torch.load(
                        output_dir / "best.pt", weights_only=True, map_location="cpu"
                    )
                    conditioning_dim = (3 if scalars else 0) + (7 if sequences else 0)
                    channels = 3 if grids else 0
                    restored = build_model(
                        shape, channels, 2, cfg.arch.afno, cfg.film, conditioning_dim
                    ).eval()
                    restored.load_state_dict(checkpoint["model_state_dict"])
                    if dims == 3:
                        backbone = restored.backbone if enabled else restored
                        self.assertIsInstance(backbone, AFNO3D)
                    if enabled or dims == 3:
                        self.assertEqual(checkpoint["in_channels"], channels)
                        self.assertEqual(checkpoint["out_channels"], 2)
                        self.assertEqual(checkpoint["conditioning_dim"], conditioning_dim)
                    with np.load(output_dir / "predictions.npz") as exported:
                        self.assertEqual(exported["predictions"].shape, (2, 2, *shape))
                        self.assertTrue(np.isfinite(exported["predictions"]).all())
                        inputs = {}
                        if grids:
                            self.assertEqual(exported["inputs"].shape, (2, 3, *shape))
                            inputs["grid"] = torch.from_numpy(exported["inputs"])
                        else:
                            self.assertNotIn("inputs", exported.files)
                        if enabled:
                            self.assertEqual(
                                exported["conditioning"].shape, (2, conditioning_dim)
                            )
                            inputs["conditioning"] = torch.from_numpy(exported["conditioning"])
                            self.assertEqual(
                                json.loads(str(exported["conditioning_layout"])),
                                checkpoint["conditioning_layout"],
                            )
                        else:
                            self.assertNotIn("conditioning", exported.files)
                        with torch.no_grad():
                            predicted = train._predict(
                                restored, inputs if enabled else inputs["grid"],
                                torch.device("cpu"),
                            )
                        np.testing.assert_allclose(
                            (predicted.numpy() + 1) / 2, exported["predictions"],
                            rtol=1e-5, atol=1e-6,
                        )


if __name__ == "__main__":
    unittest.main()
