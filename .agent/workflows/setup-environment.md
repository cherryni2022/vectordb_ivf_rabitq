---
description: Setup Python virtual environment and install dependencies using uv
---

# Setup Python Environment

This workflow sets up the Python virtual environment and installs all project dependencies using uv.

## Prerequisites

- uv is available at `/Users/niwen/anaconda3/bin/uv`
- Python 3.12+ is required (see `pyproject.toml`)

## Steps

1. Navigate to the project directory
```bash
cd /Users/niwen/PycharmProjects/coleam00_proj/study_vectordb
```

// turbo
2. Install Python 3.12 using uv (if not already installed)
```bash
/Users/niwen/anaconda3/bin/uv python install 3.12
```

3. Remove existing virtual environment (if any)
```bash
rm -rf .venv
```

// turbo
4. Create virtual environment with Python 3.12
```bash
/Users/niwen/anaconda3/bin/uv venv .venv --python 3.12
```

// turbo
5. Activate the virtual environment
```bash
source .venv/bin/activate
```

// turbo
6. Sync dependencies from pyproject.toml (including dev dependencies)
```bash
/Users/niwen/anaconda3/bin/uv sync --dev
```

## Verification

After setup, verify the environment:

// turbo
7. Check Python version
```bash
python --version
```

// turbo
8. List installed packages
```bash
pip list
```

// turbo
9. Run tests to verify everything works
```bash
pytest tests/ -v
```

## Alternative: Using pip directly

If you prefer not to use uv for dependency management:

```bash
# Create venv
/Users/niwen/anaconda3/bin/python -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```
