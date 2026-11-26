# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

if __package__:
    from . import data
    from .model import build_model
else:
    import data
    from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ModelInputs = torch.Tensor | Mapping[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to the training YAML file")
    parser.add_argument("--config_dir", "--config-dir", help="Directory containing the training YAML file")
    parser.add_argument("--config_name", "--config-name", default="config_afno")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Base directory for relative data paths (default depends on config selection)",
    )
    parser.add_argument("--max-steps", type=int, help="Override the configured maximum training steps")
    parser.add_argument(
        "--validation-frequency",
        type=int,
        help="Override how often to evaluate the validation dataset",
    )
    parser.add_argument("--train-batch-size", type=int, help="Override the configured training batch size")
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        help="Override the configured validation batch size",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the configured output directory")
    return parser.parse_args()


def _normalize_key_specs(raw_specs) -> list[list[float | str]]:
    if raw_specs is None:
        return []
    specs = (
        OmegaConf.to_container(raw_specs, resolve=True)
        if OmegaConf.is_config(raw_specs) else raw_specs
    )
    if not isinstance(specs, (list, tuple)):
        raise ValueError("Dataset specifications must be [name, lower, upper] lists")
    if not specs:
        return []
    if not isinstance(specs[0], (list, tuple)):
        specs = [specs]
    for spec in specs:
        if (
            not isinstance(spec, (list, tuple)) or len(spec) != 3
            or not isinstance(spec[0], str) or not spec[0]
        ):
            raise ValueError(f"Invalid dataset specification: {spec}")
    return [list(spec) for spec in specs]


def _normalize_arrays(
    arrays: Mapping[str, np.ndarray], key_specs: Sequence[Sequence[float | str]]
) -> dict[str, np.ndarray]:
    normalized = dict(arrays)
    for name, lower, upper in key_specs:
        normalized[str(name)] = data.normalize_array(
            arrays[str(name)], str(name), float(lower), float(upper)
        )
    return normalized


def _grid(inputs: ModelInputs) -> torch.Tensor | None:
    return inputs if isinstance(inputs, torch.Tensor) else inputs.get("grid")


def _predict(model: nn.Module, inputs: ModelInputs, device: torch.device) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        return model(inputs.to(device))
    grid = _grid(inputs)
    return model(
        grid.to(device) if grid is not None else None, inputs["conditioning"].to(device)
    )


def _training_batches(dataloader: DataLoader):
    while True:
        yield from dataloader


def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            predictions = _predict(model, inputs, device)
            total_loss += loss_fn(predictions, targets.to(device)).item() * len(targets)
    return total_loss / len(dataloader.dataset)


def _save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    output_range: Sequence[float | str],
    output_path: Path,
    device: torch.device,
    examples: int,
    conditioning_layout: Sequence[dict] = (),
) -> None:
    lower, upper = float(output_range[1]), float(output_range[2])
    model.eval()
    inputs, targets = next(iter(dataloader))
    with torch.no_grad():
        predictions = _predict(model, inputs, device).cpu()
    count = min(examples, len(targets))
    denormalize = lambda values: (values + 1.0) * (upper - lower) / 2.0 + lower
    extra = {}
    if not isinstance(inputs, torch.Tensor):
        extra = {
            "conditioning": inputs["conditioning"][:count].numpy(),
            "conditioning_layout": np.asarray(json.dumps(list(conditioning_layout))),
        }
    grid = _grid(inputs)
    if grid is not None:
        extra["inputs"] = grid[:count].numpy()
    np.savez(
        output_path,
        targets=denormalize(targets[:count]).numpy(),
        predictions=denormalize(predictions[:count]).numpy(),
        **extra,
    )


def main() -> None:
    args = parse_args()
    config = getattr(args, "config", None)
    if config is not None:
        config_path = Path(config).expanduser().resolve()
        default_data_root = Path.cwd()
    else:
        config_dir = Path(getattr(args, "config_dir", None) or "configs")
        if not config_dir.is_absolute():
            config_dir = PROJECT_ROOT / config_dir
        config_path = config_dir / Path(
            getattr(args, "config_name", "config_afno")
        ).with_suffix(".yaml")
        default_data_root = PROJECT_ROOT
    data_root = getattr(args, "data_root", None) or default_data_root
    data_root = Path(data_root).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    film_config = cfg.get("film", {})
    film_enabled = film_config.get("enabled", False)
    if not isinstance(film_enabled, bool):
        raise ValueError("film.enabled must be a boolean")

    train_batch_size = getattr(args, "train_batch_size", None)
    train_batch_size = train_batch_size if train_batch_size is not None else int(cfg.batch_size.grid)
    validation_batch_size = (
        getattr(args, "validation_batch_size", None)
        if getattr(args, "validation_batch_size", None) is not None
        else int(cfg.batch_size.validation)
    )
    validation_frequency = (
        getattr(args, "validation_frequency", None)
        if getattr(args, "validation_frequency", None) is not None
        else int(cfg.training.rec_validation_freq)
    )
    max_steps = getattr(args, "max_steps", None)
    max_steps = max_steps if max_steps is not None else int(cfg.training.max_steps)
    if validation_frequency <= 0:
        raise ValueError("Validation frequency must be greater than zero")
    if max_steps <= 0:
        raise ValueError("Maximum training steps must be greater than zero")
    if train_batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero")

    input_key_specs = _normalize_key_specs(cfg.custom.data.get("input_keys"))
    output_key_specs = _normalize_key_specs(cfg.custom.data.output_key)
    scalar_specs = _normalize_key_specs(cfg.custom.data.get("scalar_keys"))
    time_series_specs = _normalize_key_specs(cfg.custom.data.get("time_series_keys"))
    if len(output_key_specs) != 1:
        raise ValueError("Configure exactly one spatial output key")
    if not film_enabled and (scalar_specs or time_series_specs):
        raise ValueError("Scalar/time-series input keys require enabled FiLM")
    if not (input_key_specs or scalar_specs or time_series_specs):
        raise ValueError("Configure at least one spatial, scalar or time-series input")
    if film_enabled and not (scalar_specs or time_series_specs):
        raise ValueError("Enabled FiLM requires scalar_keys or time_series_keys")
    image_shape = tuple(cfg.arch.afno.img_shape)
    if len(image_shape) not in (2, 3):
        raise ValueError("AFNO img_shape must define 2 or 3 spatial dimensions")
    data_paths = [
        path if path.is_absolute() else data_root / path
        for path in map(Path, cfg.custom.data.files)
    ]
    missing_paths = [path for path in data_paths if not path.is_file()]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Configured training data files are missing:\n{missing}")
    dataset = data.HDF5GridDataset(
        data_paths, input_key_specs, output_key_specs[0],
        scalar_specs, time_series_specs, len(image_shape),
    )
    train_set, validation_set = data.split_train_validation_dataset(
        dataset,
        float(cfg.custom.data.train_fraction),
        int(cfg.custom.data.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=validation_batch_size)
    sample_inputs, sample_targets = train_set[0]
    sample_grid = _grid(sample_inputs)
    if tuple(sample_targets.shape[1:]) != image_shape:
        raise ValueError("Configured AFNO img_shape does not match the dataset grid dimensions")
    in_channels = sample_grid.shape[0] if sample_grid is not None else 0
    conditioning_dim = (
        sample_inputs["conditioning"].numel()
        if not isinstance(sample_inputs, torch.Tensor) else 0
    )
    conditioning_layout = train_set.dataset.conditioning_layout
    model = build_model(
        cfg.arch.afno.img_shape,
        in_channels,
        sample_targets.shape[0],
        cfg.arch.afno,
        film_config,
        conditioning_dim,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: float(cfg.scheduler.decay_rate) ** (step / float(cfg.scheduler.decay_steps)),
    )
    loss_fn = nn.MSELoss()
    output_dir_arg = getattr(args, "output_dir", None)
    output_dir = output_dir_arg if output_dir_arg is not None else Path(cfg.custom.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / cfg.custom.output.checkpoint_name
    prediction_path = output_dir / cfg.custom.output.predictions_name
    visualize_epochs = cfg.custom.output.get("visualize_each_epoch", False)
    if not isinstance(visualize_epochs, bool):
        raise ValueError("custom.output.visualize_each_epoch must be a boolean")
    if visualize_epochs and int(cfg.custom.output.examples_to_plot) <= 0:
        raise ValueError("Epoch visualizations require positive examples_to_plot")
    if visualize_epochs:
        if __package__:
            from .plotting import save_epoch_plots
        else:
            from plotting import save_epoch_plots
    epoch_history = []
    epoch_loss_sum = torch.zeros((), device=device)
    epoch_samples = 0
    log_frequency = int(cfg.training.get("log_freq", 0))
    if log_frequency < 0:
        raise ValueError("training.log_freq must be nonnegative")
    best_validation_loss = float("inf")

    for step, (inputs, targets) in enumerate(_training_batches(train_loader), start=1):
        model.train()
        optimizer.zero_grad()
        predictions = _predict(model, inputs, device)
        loss = loss_fn(predictions, targets.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if visualize_epochs:
            epoch_loss_sum += loss.detach() * len(targets)
            epoch_samples += len(targets)
        epoch_finished = step % len(train_loader) == 0
        validate = (
            step % validation_frequency == 0 or step == max_steps
            or (visualize_epochs and epoch_finished)
        )

        if (
            log_frequency and step % log_frequency == 0
            and not validate
        ):
            print(f"step={step} train_mse={loss.item():.6e}", flush=True)

        if validate:
            validation_loss = _evaluate(model, validation_loader, device)
            print(f"step={step} train_mse={loss.item():.6e} validation_mse={validation_loss:.6e}", flush=True)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                checkpoint = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_mse": validation_loss,
                }
                if film_enabled or len(image_shape) == 3:
                    checkpoint.update(
                        config=OmegaConf.to_container(cfg, resolve=True),
                        conditioning_layout=conditioning_layout,
                        conditioning_dim=conditioning_dim,
                        in_channels=in_channels,
                        out_channels=sample_targets.shape[0],
                    )
                torch.save(checkpoint, checkpoint_path)
                _save_predictions(
                    model,
                    validation_loader,
                    output_key_specs[0],
                    prediction_path,
                    device,
                    int(cfg.custom.output.examples_to_plot),
                    conditioning_layout,
                )
            if visualize_epochs and epoch_finished:
                epoch = step // len(train_loader)
                epoch_history.append({
                    "epoch": epoch, "step": step,
                    "train_mse": epoch_loss_sum.item() / epoch_samples,
                    "validation_mse": validation_loss,
                })
                epoch_dir = output_dir / "epochs" / f"epoch_{epoch:04d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                epoch_predictions = epoch_dir / "predictions.npz"
                _save_predictions(
                    model, validation_loader, output_key_specs[0], epoch_predictions,
                    device, int(cfg.custom.output.examples_to_plot), conditioning_layout,
                )
                save_epoch_plots(
                    epoch_predictions, epoch_history, output_key_specs[0], epoch_dir,
                )
                print(f"epoch={epoch} visualizations={epoch_dir}", flush=True)
                epoch_loss_sum.zero_()
                epoch_samples = 0

        if step == max_steps:
            break

if __name__ == "__main__":
    main()