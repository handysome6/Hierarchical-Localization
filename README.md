# Stereo Depth Pipeline

A visual odometry pipeline that combines hloc's state-of-the-art feature matching with depth-based pose estimation and GTSAM bundle adjustment for accurate camera trajectory estimation and point cloud concatenation.

## Features

- **SuperPoint + LightGlue**: State-of-the-art learned feature extraction and matching
- **Depth-based PnP**: Uses pre-computed depth maps for accurate 3D-2D pose estimation
- **Exhaustive Pairing**: Matches all frame pairs for dense constraint graphs (configurable)
- **GTSAM Bundle Adjustment**: Global pose optimization with relative pose constraints
- **Point Cloud Concatenation**: Merge individual point clouds using estimated poses
- **Open3D Integration**: Efficient point cloud I/O and voxel downsampling

## Installation

### Basic Installation

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .

# Pull external feature models
git submodule update --init --recursive
```

### For Bundle Adjustment (GTSAM)

GTSAM requires Python 3.10 or 3.11 (not compatible with Python 3.13+). Create a dedicated conda environment:

```bash
# Create environment with Python 3.10
mamba create -n hloc_stereo python=3.10 -y
mamba activate hloc_stereo

# Install GTSAM
mamba install -c conda-forge gtsam -y

# Install hloc and dependencies
cd Hierarchical-Localization
pip install -e .
pip install open3d

# Pull submodules if not done
git submodule update --init --recursive
```

### Verify Installation

```python
from hloc.pipelines.StereoDepth.pipeline import GTSAM_AVAILABLE
print(f"GTSAM available: {GTSAM_AVAILABLE}")
```

## Dataset Format

Organize your data with timestamp-named folders:

```
data_dir/
├── 1705123456/              # Timestamp folder (numeric name)
│   ├── rect_left.jpg        # Rectified left image
│   ├── rect_right.jpg       # Rectified right image (optional)
│   ├── depth_meter.npy      # Depth map in meters (H x W, float32)
│   ├── cloud.ply            # Point cloud for concatenation
│   └── K.txt                # Camera intrinsics
├── 1705123457/
│   ├── rect_left.jpg
│   ├── depth_meter.npy
│   ├── cloud.ply
│   └── K.txt
└── ...
```

### K.txt Format

```
fx 0 cx 0 fy cy 0 0 1
baseline
```

Example:
```
2247.5 0 2063.5 0 2247.5 1505.5 0 0 1
0.12
```

- Line 1: 9 values representing the 3x3 intrinsic matrix in row-major order
- Line 2: Stereo baseline in meters (used for reference, not in pose estimation)

### Depth Map Format

- NumPy array saved as `.npy` file
- Shape: (H, W) matching the image dimensions
- Values: Depth in meters (float32)
- Invalid depth: Use 0 or negative values

## Usage

### Command Line Interface

Basic usage:
```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir /path/to/your/data \
    --output_dir /path/to/output
```

With Bundle Adjustment:
```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir /path/to/your/data \
    --output_dir /path/to/output \
    --ba \
    --voxel_size 0.01
```

Full options:
```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir /path/to/your/data \
    --output_dir /path/to/output \
    --ba \                        # Enable Bundle Adjustment
    --voxel_size 0.01 \           # Point cloud downsampling (meters)
    --sequential_pairs \          # Use sequential pairing only (faster)
    --max_exhaustive 50 \         # Max frames for exhaustive pairing
    --no_pcd \                    # Skip point cloud concatenation
    --verbose                     # Enable debug logging
```

### Python API

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Required on Windows

from pathlib import Path
from hloc.pipelines.StereoDepth.pipeline import run_pipeline

results = run_pipeline(
    data_dir=Path("/path/to/your/data"),
    output_dir=Path("/path/to/output"),
    concatenate_pcd=True,
    voxel_size=0.01,
    run_ba=True,
    exhaustive_pairs=True,
    max_exhaustive_frames=50,
)

print(f"Successful poses: {results['num_successful']}/{results['num_frames']-1}")
print(f"Poses: {results['poses']}")
```

### Windows Note

On Windows, you may encounter OpenMP library conflicts. Set this environment variable before running:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

Or in the command line:
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
python -m hloc.pipelines.StereoDepth.pipeline ...
```

## Configuration Options

### run_pipeline() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | Path | required | Directory containing timestamp folders |
| `output_dir` | Path | required | Output directory for results |
| `visualize` | bool | False | Show visualizations (not implemented) |
| `concatenate_pcd` | bool | True | Concatenate point clouds |
| `voxel_size` | float | 0.01 | Voxel size for downsampling (meters, 0 to disable) |
| `run_ba` | bool | False | Run Bundle Adjustment with GTSAM |
| `exhaustive_pairs` | bool | True | Use exhaustive pairing for BA |
| `max_exhaustive_frames` | int | 50 | Fall back to sequential if more frames |

### Feature Extraction Config

Default SuperPoint configuration (can be modified in `pipeline.py`):

```python
EXTRACT_CONF = {
    "output": "feats-superpoint-n4096-r1600",
    "model": {
        "name": "superpoint",
        "nms_radius": 3,
        "max_keypoints": 4096,
    },
    "preprocessing": {
        "grayscale": True,
        "resize_max": 1600,
    },
}
```

### Matching Config

Default LightGlue configuration:

```python
MATCH_CONF = {
    "output": "matches-lightglue",
    "model": {
        "name": "lightglue",
        "features": "superpoint",
    },
}
```

## Output Files

The pipeline generates the following files in `output_dir`:

```
output_dir/
├── images/                  # Symlinked/copied input images
├── features.h5              # Extracted SuperPoint features
├── pairs.txt                # Image pairs for matching
├── matches.h5               # LightGlue matches
├── poses.txt                # Estimated camera poses (4x4 matrices)
├── pose_results.json        # Detailed results per frame pair
└── combined.ply             # Concatenated point cloud (if enabled)
```

### poses.txt Format

```
# image_name.jpg
T00 T01 T02 T03
T10 T11 T12 T13
T20 T21 T22 T23
T30 T31 T32 T33
```

Each pose is a 4x4 transformation matrix (world_T_camera) that transforms points from camera frame to world frame.

### pose_results.json Format

```json
[
  {
    "frame0": "1705123456.jpg",
    "frame1": "1705123457.jpg",
    "success": true,
    "num_matches": 1523,
    "num_depth_valid": 1234,
    "num_inliers": 892,
    "translation": 0.156
  },
  ...
]
```

## Pipeline Steps

1. **Image Preparation**: Copy images to output directory with standardized names
2. **Feature Extraction**: Extract SuperPoint features for all images
3. **Pair Generation**: Create exhaustive or sequential image pairs
4. **Feature Matching**: Match features using LightGlue
5. **Sequential PnP**: Estimate initial poses using depth-based PnP
6. **BA Observations**: Collect observations from all matched pairs
7. **Bundle Adjustment**: Optimize poses with GTSAM (optional)
8. **Point Cloud Concatenation**: Merge clouds using optimized poses (optional)

## Pairing Strategies

### Sequential Pairing (--sequential_pairs)
- Matches consecutive frames only: (0,1), (1,2), (2,3), ...
- **Pairs**: n-1 for n frames
- **Pros**: Fast, minimal computation
- **Cons**: No loop closure, drift accumulates

### Exhaustive Pairing (default)
- Matches all frame combinations: (0,1), (0,2), ..., (n-2,n-1)
- **Pairs**: n(n-1)/2 for n frames
- **Pros**: Dense constraints, automatic loop closure, reduced drift
- **Cons**: O(n²) complexity, slower for large datasets

| Frames | Sequential Pairs | Exhaustive Pairs |
|--------|------------------|------------------|
| 6 | 5 | 15 |
| 12 | 11 | 66 |
| 20 | 19 | 190 |
| 50 | 49 | 1225 |

## Examples

### Example 1: Quick Run (No BA)

```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir ./my_data \
    --sequential_pairs
```

### Example 2: Full Pipeline with BA

```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir ./my_data \
    --output_dir ./output \
    --ba \
    --voxel_size 0.005 \
    --verbose
```

### Example 3: Python Script

```python
#!/usr/bin/env python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from hloc.pipelines.StereoDepth.pipeline import run_pipeline, GTSAM_AVAILABLE

print(f"GTSAM available: {GTSAM_AVAILABLE}")

results = run_pipeline(
    data_dir=Path("C:/data/office_scan"),
    output_dir=Path("C:/data/office_scan/hloc_output"),
    concatenate_pcd=True,
    voxel_size=0.01,
    run_ba=GTSAM_AVAILABLE,
    exhaustive_pairs=True,
)

print(f"Results: {results['num_successful']}/{results['num_frames']-1} successful")

# Access poses
for img_name, pose in results['poses'].items():
    print(f"{img_name}: t = {pose[:3, 3]}")
```

## Troubleshooting

### "GTSAM not available"
- GTSAM requires Python 3.10 or 3.11
- Install via conda: `mamba install -c conda-forge gtsam`

### "OpenMP library conflict" (Windows)
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

### "Too few matches" or "PnP failed"
- Check image quality and overlap between consecutive frames
- Ensure depth maps are valid (positive values in meters)
- Try increasing `max_keypoints` in EXTRACT_CONF

### "Low inlier ratio"
- Images may have large viewpoint changes
- Try capturing with smaller motion between frames
- The pipeline uses 8.0 pixel reprojection threshold for high-res images

### Memory issues with large datasets
- Use `--sequential_pairs` to reduce matching pairs
- Reduce `max_keypoints` in configuration
- Process in batches if necessary

## Technical Details

### Coordinate Conventions

- **Pose Convention**: `poses[img]` is `T_world_camera` (transforms camera points to world)
- **Depth Convention**: Depth values in meters along the camera Z-axis
- **Point Cloud**: Points in camera coordinate frame before transformation

### PnP Estimation

- Uses OpenCV's `solvePnPRansac` with P3P method
- Reprojection error threshold: 8.0 pixels (suitable for high-res images)
- Minimum inliers: 6 points
- Inlier ratio threshold: 15%

### Bundle Adjustment

- Uses GTSAM's `BetweenFactorPose3` for relative pose constraints
- Noise model scales inversely with number of inliers
- First pose is fixed with tight prior
- Levenberg-Marquardt optimizer with 100 max iterations

## License

This pipeline is part of the hloc toolbox. See LICENSE file for details.

## Acknowledgments

- [hloc](https://github.com/cvg/Hierarchical-Localization) - Hierarchical Localization toolbox
- [SuperPoint](https://arxiv.org/abs/1712.07629) - Self-supervised interest point detection
- [LightGlue](https://github.com/cvg/LightGlue) - Fast feature matching
- [GTSAM](https://gtsam.org/) - Georgia Tech Smoothing and Mapping library
- [Open3D](http://www.open3d.org/) - 3D data processing library
