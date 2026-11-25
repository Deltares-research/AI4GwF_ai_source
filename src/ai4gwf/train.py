# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import data
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to the training YAML file")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative paths in the YAML file (default: current directory)",
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
    specs = OmegaConf.to_container(raw_specs, resolve=True)
    if specs and isinstance(specs[0], (list, tuple)):
        return [list(spec) for spec in specs]
    return [list(specs)]


def _normalize_arrays(
    arrays: Mapping[str, np.ndarray], key_specs: Sequence[Sequence[float | str]]
) -> dict[str, np.ndarray]:
    normalized = dict(arrays)
    for name, lower, upper in key_specs:
        lower, upper = float(lower), float(upper)
        if upper <= lower:
            raise ValueError(f"Invalid range for '{name}': [{lower}, {upper}]")
        normalized[str(name)] = 2.0 * (arrays[str(name)] - lower) / (upper - lower) - 1.0
    return normalized


def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            predictions = model(inputs.to(device))
            total_loss += loss_fn(predictions, targets.to(device)).item() * len(inputs)
    return total_loss / len(dataloader.dataset)


def _save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    output_range: Sequence[float | str],
    output_path: Path,
    device: torch.device,
    examples: int,
) -> None:
    lower, upper = float(output_range[1]), float(output_range[2])
    model.eval()
    inputs, targets = next(iter(dataloader))
    with torch.no_grad():
        predictions = model(inputs.to(device)).cpu()
    count = min(examples, len(inputs))
    denormalize = lambda values: (values + 1.0) * (upper - lower) / 2.0 + lower
    np.savez(
        output_path,
        inputs=inputs[:count].numpy(),
        targets=denormalize(targets[:count]).numpy(),
        predictions=denormalize(predictions[:count]).numpy(),
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = OmegaConf.load(config_path)

    train_batch_size = (
        args.train_batch_size if args.train_batch_size is not None else int(cfg.batch_size.grid)
    )
    validation_batch_size = (
        args.validation_batch_size
        if args.validation_batch_size is not None
        else int(cfg.batch_size.validation)
    )
    validation_frequency = (
        args.validation_frequency
        if args.validation_frequency is not None
        else int(cfg.training.rec_validation_freq)
    )
    max_steps = args.max_steps if args.max_steps is not None else int(cfg.training.max_steps)
    if validation_frequency <= 0:
        raise ValueError("Validation frequency must be greater than zero")
    if max_steps <= 0:
        raise ValueError("Maximum training steps must be greater than zero")
    if train_batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero")

    input_key_specs = _normalize_key_specs(cfg.custom.data.input_keys)
    output_key_specs = _normalize_key_specs(cfg.custom.data.output_key)
    input_key_names = [str(item[0]) for item in input_key_specs]
    output_key_name = str(output_key_specs[0][0])
    data_paths = [
        path if path.is_absolute() else args.data_root.expanduser().resolve() / path
        for path in map(Path, cfg.custom.data.files)
    ]
    missing_paths = [path for path in data_paths if not path.is_file()]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Configured training data files are missing:\n{missing}")
    arrays = data.load_hdf5_files(data_paths, input_key_names + [output_key_name])
    arrays = _normalize_arrays(arrays, input_key_specs + output_key_specs)
    train_set, validation_set = data.make_train_validation_datasets(
        arrays,
        input_key_names,
        output_key_name,
        float(cfg.custom.data.train_fraction),
        int(cfg.custom.data.seed),
    )
    del arrays

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=validation_batch_size)
    sample_inputs, sample_targets = train_set[0]
    model = build_model(
        cfg.arch.afno.img_shape,
        sample_inputs.shape[0],
        sample_targets.shape[0],
        cfg.arch.afno,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: float(cfg.scheduler.decay_rate) ** (step / float(cfg.scheduler.decay_steps)),
    )
    loss_fn = nn.MSELoss()
    output_dir = (
        args.output_dir if args.output_dir is not None else Path(cfg.custom.output.directory)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / cfg.custom.output.checkpoint_name
    prediction_path = output_dir / cfg.custom.output.predictions_name
    best_validation_loss = float("inf")

    train_iterator = iter(train_loader)
    for step in range(1, max_steps + 1):
        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            inputs, targets = next(train_iterator)

        model.train()
        optimizer.zero_grad()
        predictions = model(inputs.to(device))
        loss = loss_fn(predictions, targets.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % validation_frequency == 0 or step == max_steps:
            validation_loss = _evaluate(model, validation_loader, device)
            print(f"step={step} train_mse={loss.item():.6e} validation_mse={validation_loss:.6e}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "validation_mse": validation_loss,
                    },
                    checkpoint_path,
                )
                _save_predictions(
                    model,
                    validation_loader,
                    output_key_specs[0],
                    prediction_path,
                    device,
                    int(cfg.custom.output.examples_to_plot),
                )

if __name__ == "__main__":
    main()