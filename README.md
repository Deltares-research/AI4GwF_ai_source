# AI4GwF: AFNO with optional FiLM

Predict 2D spatial grids or true 3D volumes from any nonempty combination of
spatial arrays, multiple scalars/vectors, and fixed-length time-series.
The 2D backbone is NVIDIA PhysicsNeMo AFNO; the 3D backbone is a native PyTorch
volumetric AFNO. Scalars and time-series use Feature-wise Linear Modulation
(FiLM). Spatial input arrays are optional; the output is always spatial.

## Installation and training

Use Python 3.10+ and a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install matching PyTorch and torchvision builds for your CPU/CUDA environment
following https://pytorch.org/get-started/locally/, then the runtime dependencies.
For example, the tested CPU setup is:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The FiLM adapter targets PhysicsNeMo 1.3.0, pinned because it traverses AFNO's
exposed components. CPU validation uses PyTorch 2.8.0 and torchvision 0.23.0.
No GPU or PhysicsNeMo optional extras are required for these models.
For GPU training, install matching CUDA PyTorch/torchvision wheels instead of
the CPU wheels (for example, the same versions with
`--index-url https://download.pytorch.org/whl/cu128`). The trainer automatically
selects CUDA when available. Use an isolated environment and verify
`torch.cuda.is_available()` before starting.

Edit [the grid-only config](configs/config_afno.yaml) or
[the mixed-input FiLM config](configs/config_afno_film.yaml) for your data.
[The conditioning-only config](configs/config_afno_conditioning.yaml) needs no
spatial inputs; [the volume config](configs/config_afno_3d.yaml) demonstrates
true 3D outputs. Run from the repository root:

```bash
# Plain AFNO (also the default config)
python src/ai4gwf/train.py --config_name config_afno

# AFNO with FiLM
python src/ai4gwf/train.py --config_name config_afno_film

# Scalars/time-series only, predicting a 2D grid
python src/ai4gwf/train.py --config_name config_afno_conditioning

# True 3D volume prediction
python src/ai4gwf/train.py --config_name config_afno_3d

# Equivalent package entry point
PYTHONPATH=src python -m ai4gwf.train --config_name config_afno_film
```

Use `--config_dir /path/to/configs` for another config directory. Relative config
and HDF5 paths resolve against the repository root. Relative output directories
resolve against the working directory. Example data paths are placeholders; no
training data is bundled.

## HDF5 data contract

Each selected key identifies a numeric HDF5 dataset. `N` is the sample axis;
record `i` must refer to the same example across every selected dataset.

| Config key | Supported dataset shapes | Role |
| --- | --- | --- |
| `custom.data.input_keys` (2D) | `(N,H,W)`, `(N,C,H,W)` | Optional channel-stacked spatial inputs |
| `custom.data.output_key` (2D) | `(N,H,W)`, `(N,Cout,H,W)` | One spatial target dataset |
| `custom.data.input_keys` (3D) | `(N,D,H,W)`, `(N,C,D,H,W)` | Optional channel-stacked volumetric inputs |
| `custom.data.output_key` (3D) | `(N,D,H,W)`, `(N,Cout,D,H,W)` | One volumetric target dataset |
| `custom.data.scalar_keys` | `(N,)`, `(N,F)` | Optional scalar/vector conditioning |
| `custom.data.time_series_keys` | `(N,T)`, `(N,T,F)` | Optional time-series conditioning |

For example, the FiLM config expects `permeability` and `head` grids of size
32 x 32, a `pumping_rate` scalar/vector, and `rainfall_history` and
`rainfall_forecast` sequences, all sharing `N`. Different sequence keys may have
different lengths; each key's length must be fixed across files and inference.

`arch.afno.img_shape` declares the spatial axes: `[H,W]` selects PhysicsNeMo
AFNO, while `[D,H,W]` selects volumetric AFNO. Thus a `(N,C,H,W)` dataset in a
2D configuration is not a true volume. Each model has a fixed output grid;
spatial inputs, when present, must have the same dimensionality and resolution
as the target. Mixing 2D maps with 3D volume inputs in one model, or changing
input/output resolution, is not supported.

Every key specification is `[dataset_name, lower, upper]`. Values are mapped
linearly to `[-1,1]` using those bounds, without clipping. Choose finite bounds
with `lower < upper`; choose them from training/domain knowledge, not validation
statistics. The same ranges normalize grids, scalars and sequences. Predictions
and targets are exported in original target units.

Files concatenate logically along `N`; non-sample shapes must match for each key. Grids
must share the configured `img_shape`, divisible by `patch_size`. At least two
samples are required. Missing keys, duplicate roles, empty dimensions, mismatched
sample counts, unsupported shapes, and non-finite data raise explicit errors.

## Low-memory loading

Training uses `HDF5GridDataset` automatically: startup reads only file metadata,
then each sample's selected datasets are read and normalized on demand. Nothing
materializes the full HDF5 arrays. Shapes, key roles and normalization bounds are
validated at startup; non-finite values are rejected with the file, key and sample
index when read.

File handles open and close within each sample read, so they cannot leak or be
shared across DataLoader workers. The trainer uses zero background workers and
does not cache batches between epochs; each epoch reads fresh samples and
reshuffles the training split.

Loader memory is primarily the current batch, temporary sample arrays, and HDF5
chunk buffers, rather than the entire dataset. Train/validation split indices
still require a small amount of memory proportional to the sample count.
Reduce `batch_size.grid` and `batch_size.validation` (for example to `4` or `1`)
to lower RAM/VRAM further; model weights and activations still consume memory.
For compressed files, prefer sample-sized chunks such as `(1,C,H,W)` or
`(1,C,D,H,W)` rather than chunks spanning many samples.

The eager `load_hdf5_files` / `GridDataset` helpers remain available for small
in-memory datasets and compatibility, but the training entry point no longer
uses them. Streaming does not change dataset units or derive missing features:
the original Vitens source still needs a config/preprocessing appropriate for
its signed `q` and raw targets.

Provide already-aligned historical or forecast sequences. There is no automatic
windowing, resampling, timestamp interpretation, padding or missing-value masking.
You are responsible for ensuring forecast inputs are available at prediction
time and for avoiding target leakage. The existing split is random by sample;
prepare a suitable split strategy before using overlapping temporal windows or
data that require chronological validation.

## FiLM configuration and behavior

```yaml
film:
  enabled: true
  hidden_dim: 64
custom:
  data:
    scalar_keys:
      - [pumping_rate, 0.0, 1000.0]
    time_series_keys:
      - [rainfall_history, 0.0, 100.0]
```

This is a fragment; use the complete example configs for training. At least one
scalar or time-series key is required when FiLM is enabled. Select your input
combination by leaving unused key lists empty (or omitting them):

| Inputs | `input_keys` | `scalar_keys` | `time_series_keys` | FiLM |
| --- | --- | --- | --- | --- |
| Spatial arrays only | Nonempty | Empty | Empty | Disabled |
| Scalars only | Empty | Nonempty | Empty | Enabled |
| Time-series only | Empty | Empty | Nonempty | Enabled |
| Scalars + time-series | Empty | Nonempty | Nonempty | Enabled |
| Arrays + scalars | Nonempty | Nonempty | Empty | Enabled |
| Arrays + time-series | Nonempty | Empty | Nonempty | Enabled |
| All three | Nonempty | Nonempty | Nonempty | Enabled |

These combinations work for both 2D and 3D outputs. For example, to use only
scalars, set `input_keys: []` and `time_series_keys: []`, and specify any number
of scalar keys. The spatial target must still be present in the HDF5 files.

Features concatenate in configured scalar-key order, then time-series-key order.
Each sequence flattens in row-major order (`time`, then feature), preserving
ordering. A shared MLP encodes this vector, and a separate projection generates
channel-wise gamma and beta after each AFNO block:

```text
h = (1 + gamma(conditioning)) * h + beta(conditioning)
```

Parameters broadcast over the spatial patch grid. With spatial inputs,
zero-initialized projections make FiLM initially identical to its own plain
AFNO backbone. With conditioning-only inputs, initial predictions are independent
of conditioning. The encoder starts receiving gradients after those projections
update. The AFNO spectral layers,
patch embedding and output head are reused unchanged for spatial-input models.

With no spatial input, AFNO's learned positional embedding serves as a shared
spatial feature seed. FiLM turns that seed into sample-specific features and
the AFNO blocks/head produce the output grid. No fake spatial input dataset is
required, and the unused input patch embedding is frozen and bypassed.

Set `film.enabled: false`, or omit `film`, for grid-only training. This returns
ordinary PhysicsNeMo AFNO in 2D or the volumetric backbone in 3D. **Scalar and
time-series key lists must then be empty or omitted**; configurations requesting
those inputs with FiLM disabled fail explicitly rather than silently ignoring
data. The same entry point handles every combination. Variable-length sequences
and non-spatial targets are not supported.

## Volumetric AFNO

The 3D model embeds volume patches using `Conv3d`. Every spectral block applies
`rfftn`/`irfftn` across depth, height, and width, with block-diagonal complex
channel mixing, real/imaginary ReLU, and soft-thresholding. It uses the full
3D spectrum, residual connections, channel-wise layer normalization and an MLP.
The linear head reconstructs all three spatial axes. Depth is not treated as
channels or as independent 2D slices.

For 3D, `arch.afno.sparsity_threshold` is optional (default `0.01`) and must
be finite and nonnegative. Volume dimensions must be divisible by their patch
sizes. Memory grows with the volume's patch count; start with smaller volumes
and batches. This native backbone is CPU-tested, not a PhysicsNeMo-provided 3D
model, and does not claim its ONNX/distributed/optimized-backend support.

## Outputs and checkpoints

Training saves the best validation checkpoint and predictions from the first
validation batch, limited by `examples_to_plot`. Plain 2D AFNO keeps the original
checkpoint keys/state-dict layout and NPZ fields `inputs`, `targets`,
`predictions`. Inputs remain normalized.

Set `training.log_freq` to a positive step interval (for example `100`) to
stream training loss between validations. Missing or zero keeps validation-only
logging. `training.rec_validation_freq` controls validation/checkpoint cadence.

Set `custom.output.visualize_each_epoch: true` to validate and render at every
complete training epoch, independently of the validation step interval.
`examples_to_plot` must then be positive and limits examples from the first
validation batch. Each epoch is retained under `epochs/epoch_0001/`, etc.,
with `predictions.npz` and a ground-truth/prediction/signed-error PNG per output
channel, even when validation does not improve. Volumes show the central depth
slice. Ground truth and prediction share color limits, and predictions are not
clipped. Plots use stored target units (log units if the dataset was transformed),
not an automatic inverse transform. `epochs/loss_curve.png` and
`epochs/loss_history.json` track sample-weighted epoch training MSE and validation
MSE in normalized units. Partial final epochs do not produce an epoch plot.
The option defaults to false; existing best-checkpoint exports are unchanged.

To prepare the original full Vitens scalar source without changing it:

```bash
python scripts/preprocess_vitens_film.py \
  --source /path/to/vitens_total_drawdown.h5 \
  --output /path/to/vitens_total_film_log.h5
```

This streaming copy preserves raw datasets and adds `log10_negative_q = log10(-q)`
and `groundtruth_log10 = log10(clip(groundtruth, -10000, -1) / -1000)`, matching
the original barcode preprocessing. It rejects nonnegative/non-finite pumping,
positive/non-finite drawdown, and existing output files. Spatial datasets use
sample-sized compressed chunks. Select the new pumping key in `scalar_keys`
and `[groundtruth_log10, -3.0, 1.0]` as `output_key`. The target clipping is
intentional and lossy: predictions in these log units cannot recover values
outside the original clipping range.

FiLM and all 3D checkpoints additionally contain the resolved `config`,
`conditioning_dim`, and ordered `conditioning_layout` (key, kind, non-sample
shape), plus `in_channels` and `out_channels`. `in_channels: 0` identifies a
conditioning-only model. Construct the same model with `build_model`, the saved
`config.arch.afno.img_shape`, channel counts, AFNO/FiLM settings and
`conditioning_dim` before loading `model_state_dict`. Maintain the saved key order, shapes, and
normalization during inference. Conditioned NPZ files additionally contain the
normalized flattened `conditioning` array and a JSON `conditioning_layout`
string, readable without NumPy pickle support. With no spatial input, the NPZ
contains no `inputs` field; `conditioning` records the actual model inputs.
Exports are channels-first: `(N,Cout,H,W)` or `(N,Cout,D,H,W)`.

A FiLM checkpoint is not interchangeable with a plain AFNO checkpoint;
disabling FiLM does not convert it. Existing plain AFNO checkpoints remain
compatible with disabled mode. Automatic resume/conversion is not implemented.

## Decoding existing Vitens barcodes

The Vitens conversion helper recovers 20 pre-barcode parameters from the
constant vertical stripes in `c` (7), `kd` (8), and `rest` (5):

```bash
python scripts/decode_vitens_barcodes.py \
  --source /path/to/vitens_barcode_train.h5 \
  --config /path/to/config_vitens.yaml \
  --output /path/to/vitens_scalar_train.h5
```

It uses the original config's barcode normalization bounds, checks every
stripe, handles remainder columns, and creates one `(N,)` dataset per
parameter. Row, column and layer are stored as `int32`; the other 17 values
are `float64` log10 factors or log10 pumping magnitude. It leaves the target
unchanged and refuses to overwrite existing files.

Use these keys as `scalar_keys` with `input_keys: []`. Choose individual
normalization bounds for indices and log parameters rather than reusing the
combined `rest` bounds. Check target units: the historical Vitens notebook
stores `log10(clip(drawdown, -10000, -1) / -1000)` with range `[-3,1]`;
these targets must not be normalized as raw drawdown.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests create temporary HDF5 files, exercise actual PhysicsNeMo AFNO and native
volumetric AFNO on CPU, and run training/evaluation/export/checkpoint round-trips
for all seven input combinations in both spatial dimensions. They also check
3D patch ordering, nonlocal Fourier coupling on every spatial axis, spectral
gradient consistency, sparsity and conditioning-only seed learning.
