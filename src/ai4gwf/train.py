# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from itertools import cycle
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import data as data
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--config_name", default="config_vitens")
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
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir
    config_path = config_dir / Path(args.config_name).with_suffix(".yaml")
    cfg = OmegaConf.load(config_path)

    input_key_specs = _normalize_key_specs(cfg.custom.data.input_keys)
    output_key_specs = _normalize_key_specs(cfg.custom.data.output_key)
    input_key_names = [str(item[0]) for item in input_key_specs]
    output_key_name = str(output_key_specs[0][0])
    data_paths = [
        path if path.is_absolute() else PROJECT_ROOT / path
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=int(cfg.batch_size.grid), shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=int(cfg.batch_size.validation))
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
    output_dir = Path(cfg.custom.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / cfg.custom.output.checkpoint_name
    prediction_path = output_dir / cfg.custom.output.predictions_name
    validation_frequency = int(cfg.training.rec_validation_freq)
    best_validation_loss = float("inf")

    for step, (inputs, targets) in enumerate(cycle(train_loader), start=1):
        model.train()
        optimizer.zero_grad()
        predictions = model(inputs.to(device))
        loss = loss_fn(predictions, targets.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % validation_frequency == 0 or step == int(cfg.training.max_steps):
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

        if step == int(cfg.training.max_steps):
            break


if __name__ == "__main__":
    main()