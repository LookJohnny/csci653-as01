# DarkHorse Packaging Guide

This guide explains how to package and distribute DarkHorse as a reusable application.

## Package Structure

```
darkhorse/
├── setup.py                      # Package configuration
├── MANIFEST.in                   # Files to include in distribution
├── requirements.txt              # Dependencies
├── config.example.yaml           # Example configuration
├── README.md                     # Main documentation
├── LICENSE                       # License file
├── main.py                       # CLI entry point
├── amazon_unify_pipeline.py     # Data processing pipeline
├── build_weekly_dataset.py      # Weekly aggregation
├── train_transformer.py         # Transformer training
├── forecast_pipeline.py         # Forecast ensemble
├── forecast_ops/                # Forecasting modules
│   ├── __init__.py
│   ├── autots_model.py
│   ├── tft_model.py
│   ├── ensemble.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── plots.py
├── tests/                       # Unit tests
│   ├── __init__.py
│   └── test_*.py
├── *.slurm                      # Slurm job scripts
└── *.sh                         # Shell scripts
```

## Installation Methods

### Method 1: Editable Install (Development)

Best for active development - changes to code are immediately reflected:

```bash
cd /path/to/darkhorse
pip install -e .
```

### Method 2: Standard Install

Install from local directory:

```bash
pip install /path/to/darkhorse
```

### Method 3: Install from Git

Direct installation from repository:

```bash
pip install git+https://github.com/yourusername/darkhorse.git
```

### Method 4: Build and Install Wheel

Create distributable package:

```bash
# Build wheel
python setup.py bdist_wheel

# Install wheel
pip install dist/darkhorse_forecasting-1.0.0-py3-none-any.whl
```

## CLI Commands After Installation

Once installed, you can use these commands from anywhere:

```bash
# Interactive menu
darkhorse

# Specific pipelines
darkhorse-unify --category Electronics --work-dir /scratch/$USER/stage
darkhorse-build-weekly --input reviews.csv --out weekly_panel.csv
darkhorse-train --data weekly_panel.csv --out model_output
darkhorse-forecast --dataset weekly_panel.csv --series-col parent_asin
```

## Distribution

### Create Source Distribution

```bash
python setup.py sdist
```

This creates `dist/darkhorse-forecasting-1.0.0.tar.gz`

### Create Wheel Distribution

```bash
pip install wheel
python setup.py bdist_wheel
```

This creates `dist/darkhorse_forecasting-1.0.0-py3-none-any.whl`

### Upload to PyPI (Optional)

```bash
# Install twine
pip install twine

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Docker Container (Optional)

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ git \\
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Entry point
ENTRYPOINT ["darkhorse"]
CMD ["--help"]
```

### Build Docker Image

```bash
docker build -t darkhorse:1.0.0 .
```

### Run Docker Container

```bash
docker run -it \\
  -v /path/to/data:/data \\
  -v /path/to/output:/output \\
  darkhorse:1.0.0 --interactive
```

## Singularity Container (HPC)

For HPC environments that use Singularity:

### Create Singularity Definition File

```singularity
Bootstrap: docker
From: python:3.10-slim

%files
    . /opt/darkhorse

%post
    cd /opt/darkhorse
    pip install -r requirements.txt
    pip install -e .

%environment
    export PYTHONUNBUFFERED=1

%runscript
    exec darkhorse "$@"
```

### Build Singularity Image

```bash
sudo singularity build darkhorse.sif darkhorse.def
```

### Run on HPC

```bash
singularity run darkhorse.sif --interactive
```

## Configuration Management

### Create User Config

```bash
# Copy example config
cp config.example.yaml ~/.darkhorse/config.yaml

# Edit your settings
vim ~/.darkhorse/config.yaml
```

### Use Config in Code

```python
import yaml
from pathlib import Path

config_path = Path.home() / ".darkhorse" / "config.yaml"
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
```

## Versioning

Update version in `setup.py`:

```python
setup(
    name="darkhorse-forecasting",
    version="1.1.0",  # Update here
    ...
)
```

Follow semantic versioning:
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## Testing Before Distribution

```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate

# Install package
pip install .

# Test CLI commands
darkhorse --help
darkhorse-unify --help
darkhorse-train --help

# Run tests
pytest tests/

# Deactivate
deactivate
```

## Troubleshooting

### Import Errors

If modules can't be found after installation:

```bash
# Reinstall in editable mode
pip uninstall darkhorse-forecasting
pip install -e .
```

### Missing Dependencies

```bash
# Install all dev dependencies
pip install -r requirements.txt
pip install pytest wheel twine
```

### Module Not Found in forecast_ops

Ensure `forecast_ops/__init__.py` exists:

```python
# forecast_ops/__init__.py
from .autots_model import *
from .tft_model import *
from .ensemble import *
from .data_utils import *
from .metrics import *
from .plots import *
```

## Deployment Checklist

- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md` (if exists)
- [ ] Run all tests: `pytest tests/`
- [ ] Update `README.md` with new features
- [ ] Build distributions: `python setup.py sdist bdist_wheel`
- [ ] Test installation in clean environment
- [ ] Tag release in git: `git tag v1.0.0`
- [ ] Push to repository: `git push --tags`
- [ ] Upload to PyPI (optional): `twine upload dist/*`

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/darkhorse/issues
- Email: your.email@example.com