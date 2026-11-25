# AI4GwF AI source

## Install

The project is an installable Python package. Create an environment and install it
from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The package installs the `ai4gwf-train` command. PyTorch and NVIDIA PhysicsNeMo
are runtime dependencies; use a PyTorch installation compatible with your
hardware if your environment needs a specific CUDA build.

## Train from a YAML config

Run the command from the directory that should serve as the base for relative
data paths in the YAML file. For example, with the Vitens config and data
directories as siblings:

```bash
cd /home/nuttall/vitens
ai4gwf-train --config config/config_vitens.yaml --data-root . --output-dir outputs
```

The config's `custom.data.files` entries are resolved under `--data-root`
unless they are absolute paths. Outputs are written to the configured output
directory by default. Override paths and training sizes for a small smoke run:

```bash
ai4gwf-train \
  --config config/config_vitens.yaml \
  --data-root . \
  --max-steps 1 \
  --validation-frequency 1 \
  --train-batch-size 1 \
  --validation-batch-size 1 \
  --output-dir /tmp/ai4gwf-smoke-output
```
