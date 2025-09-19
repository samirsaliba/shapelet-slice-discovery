# GENDIS Environment Setup

This repository uses conda for reliable dependency management. The `environment.yml` file contains all necessary dependencies for the GENDIS library for model performance analysis.

## Setup Instructions

Create and activate the conda environment:

```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate env

# Install the GENDIS library in development mode
pip install -e .
```

## Core Dependencies

The environment includes all essential packages for GENDIS:

- **Python 3.12**: Base runtime
- **DEAP 1.4**: Genetic algorithm framework (core for GENDIS genetic operations)
- **PyTorch 2.5.0**: Tensor operations and GPU acceleration (with CUDA 12.4)
- **tslearn 0.6.3**: Time series analysis and shapelet operations
- **dtaidistance 2.3.12**: Efficient distance calculations for time series
- **pysubgroup 0.8.0**: Subgroup discovery for the baseline approach
- **scikit-learn 1.5.2**: Machine learning utilities and evaluation
- **numpy, scipy, pandas**: Core scientific computing stack
- **matplotlib, seaborn**: Visualization for results and debugging
- **numba, llvmlite**: JIT compilation for performance optimization

## GPU Support

The environment includes CUDA 12.4 support for GPU acceleration. Ensure you have compatible NVIDIA drivers installed to take advantage of GPU speedup.

## Environment Verification

Test your installation:

```python
import torch
import deap
import tslearn
import pysubgroup
from gendis import GeneticExtractor

print("✓ All core dependencies imported successfully")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
```

## Updating the Environment

If you need to update dependencies, modify the `environment.yml` file and update your environment:

```bash
# Update existing environment
conda env update -f environment.yml

# Or recreate from scratch
conda env remove -n env
conda env create -f environment.yml
```