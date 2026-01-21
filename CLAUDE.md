# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`hloc` (Hierarchical Localization) is a modular Python toolbox for state-of-the-art 6-DoF visual localization. It combines image retrieval and feature matching for large-scale localization and Structure-from-Motion tasks.

## Build and Development Commands

### Installation
```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
git submodule update --init --recursive  # Pull external features
```

### Code Quality (CI runs these on PRs)
```bash
python -m flake8 hloc                          # Lint check
python -m isort hloc *.ipynb --check-only      # Import sort check
python -m black hloc *.ipynb --check           # Format check
```

### Auto-format Code
```bash
python -m black hloc *.ipynb
python -m isort hloc *.ipynb
```

### Running Pipelines
```bash
# Run dataset-specific pipelines
python -m hloc.pipelines.Aachen.pipeline [--outputs ./outputs/aachen]
python -m hloc.pipelines.InLoc.pipeline

# Run individual scripts
python -m hloc.extract_features --conf superpoint_aachen --images path/to/images
python -m hloc.match_features --conf superglue --pairs pairs.txt --features features.h5
```

## Code Style

- **Formatter**: black (line length 88)
- **Linter**: flake8 (max-line-length=88, ignore E203)
- **Import sorting**: isort with black profile

## Architecture

### Plugin System (BaseModel)

All feature extractors and matchers inherit from `hloc/utils/base_model.py:BaseModel`:

```python
class BaseModel(nn.Module):
    default_conf = {}      # Override with default configuration
    required_inputs = []   # Keys that must be in input data dict

    def _init(self, conf):      # Initialize model with merged config
    def _forward(self, data):   # Process data dict, return predictions dict
```

Models are dynamically loaded via `dynamic_load(root, model)` which imports `root.model` and finds the single `BaseModel` subclass.

### Adding New Extractors/Matchers

1. Create `hloc/extractors/mymethod.py` or `hloc/matchers/mymethod.py`
2. Define a class inheriting from `BaseModel`
3. Implement `default_conf`, `_init()`, and `_forward()`
4. Add a config entry to `hloc/extract_features.py` or `hloc/match_features.py`

### Data Pipeline (6 Steps)

1. **Extract features** → `extract_features.py` → HDF5 file
2. **Build SfM model** → `pairs_from_covisibility.py` + `match_features.py` + `triangulation.py`
3. **Image retrieval** → `pairs_from_retrieval.py` (global descriptors)
4. **Match queries** → `match_features.py`
5. **Localize** → `localize_sfm.py` or `localize_inloc.py`
6. **Visualize** → `visualization.py`

### HDF5 Data Format

**Features file**:
- Key: `{image_path}` → datasets: `keypoints` (Nx2), `descriptors` (DxN), `scores` (N)

**Matches file**:
- Key: `{path0}-{path1}` (slashes replaced with hyphens) → dataset: `matches0` (N indices, -1 if unmatched)

**Global descriptors file**:
- Key: `{image_path}` → dataset: `global_descriptor` (D)

### Key Directories

- `hloc/*.py` - Top-level scripts (extract_features, match_features, localize_sfm, etc.)
- `hloc/extractors/` - Feature extractor plugins (superpoint, disk, d2net, netvlad, etc.)
- `hloc/matchers/` - Matcher plugins (superglue, lightglue, loftr, nearest_neighbor, etc.)
- `hloc/pipelines/` - Dataset-specific pipelines (Aachen, InLoc, 7Scenes, Cambridge, etc.)
- `hloc/utils/` - Utilities (base_model, io, parsers, geometry, viz)
- `third_party/` - Git submodules for external models

### Configuration Pattern

Pre-defined configs in `extract_features.py` and `match_features.py` allow switching methods via string names:

```python
confs = {
    "superpoint_aachen": {
        "output": "feats-superpoint-n4096-r1024",
        "model": {"name": "superpoint", "nms_radius": 3, "max_keypoints": 4096},
        "preprocessing": {"grayscale": True, "resize_max": 1024},
    },
    # ...
}
```

## Dependencies

Core: torch>=1.1, pycolmap>=3.13.0, kornia>=0.6.11, opencv-python, h5py, numpy
