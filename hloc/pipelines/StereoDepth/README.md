# Stereo Depth Pipeline

This pipeline provides visual odometry and point cloud concatenation for stereo camera systems with pre-computed depth maps.

## Quick Start

```bash
# Basic usage
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir /path/to/data \
    --ba \
    --voxel_size 0.01

# Or from Python
from pathlib import Path
from hloc.pipelines.StereoDepth.pipeline import run_pipeline

results = run_pipeline(
    data_dir=Path("/path/to/data"),
    output_dir=Path("/path/to/output"),
    run_ba=True,
    voxel_size=0.01
)
```

## Dataset Structure

```
data_dir/
├── {timestamp1}/
│   ├── rect_left.jpg       # Rectified left image
│   ├── depth_meter.npy     # Depth map in meters (H x W)
│   ├── cloud.ply           # Point cloud
│   └── K.txt               # Camera intrinsics
├── {timestamp2}/
│   └── ...
```

### K.txt Format
```
fx 0 cx 0 fy cy 0 0 1
baseline
```

## Pipeline Overview

1. **Feature Extraction** - SuperPoint features from all images
2. **Feature Matching** - LightGlue matching (exhaustive or sequential)
3. **Sequential PnP** - Initial pose estimation with depth-based PnP
4. **BA Observations** - Collect relative poses from all matched pairs
5. **Bundle Adjustment** - GTSAM pose graph optimization (optional)
6. **Point Cloud Merge** - Concatenate clouds with optimized poses (optional)

## Key Features

### Exhaustive Pairing (Default)
- Matches all frame pairs: O(n²) complexity
- Creates dense constraint graph for BA
- Automatic loop closure detection
- Significantly reduces drift

Example: 12 frames → 66 pairs instead of 11 sequential pairs

### GTSAM Bundle Adjustment
- Uses measured relative poses from PnP
- BetweenFactorPose3 constraints
- Noise model scales with inlier count
- Typical error reduction: 80-90%

### Open3D Integration
- Efficient point cloud I/O
- Voxel downsampling
- Automatic color handling

## Command Line Options

```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir PATH              # Input data directory (required)
    --output_dir PATH            # Output directory (default: data_dir/hloc_output)
    --ba                         # Enable Bundle Adjustment
    --voxel_size 0.01            # Voxel size for downsampling (meters)
    --sequential_pairs           # Use sequential pairing only (faster)
    --max_exhaustive 50          # Max frames for exhaustive pairing
    --no_pcd                     # Skip point cloud concatenation
    --verbose                    # Enable debug logging
```

## Python API

### run_pipeline()

```python
def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    visualize: bool = False,
    concatenate_pcd: bool = True,
    voxel_size: float = 0.01,
    run_ba: bool = False,
    exhaustive_pairs: bool = True,
    max_exhaustive_frames: int = 50
) -> dict
```

**Returns:**
```python
{
    'num_frames': int,              # Total frames
    'num_successful': int,          # Successful pose estimates
    'poses': dict,                  # {image_name: 4x4 pose matrix}
    'pose_results': list            # Per-pair statistics
}
```

## Output Files

```
output_dir/
├── features.h5              # SuperPoint features
├── pairs.txt                # Image pairs for matching
├── matches.h5               # LightGlue matches
├── poses.txt                # Camera poses (4x4 matrices)
├── pose_results.json        # Detailed statistics
└── combined.ply             # Merged point cloud
```

## Configuration

### Feature Extraction (SuperPoint)

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

### Matching (LightGlue)

```python
MATCH_CONF = {
    "output": "matches-lightglue",
    "model": {
        "name": "lightglue",
        "features": "superpoint",
    },
}
```

## Performance Tuning

### Small Datasets (<20 frames)
```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir ./data \
    --ba \
    --voxel_size 0.005    # Finer voxels
```

### Medium Datasets (20-50 frames)
```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir ./data \
    --ba \
    --voxel_size 0.01     # Default
```

### Large Datasets (>50 frames)
```bash
python -m hloc.pipelines.StereoDepth.pipeline \
    --data_dir ./data \
    --sequential_pairs \  # Faster matching
    --ba \
    --voxel_size 0.02     # Coarser voxels
```

## Troubleshooting

### "PnP failed" or "Too few inliers"
- Ensure consecutive frames have sufficient overlap
- Check depth map quality (valid positive values)
- Reduce camera motion between frames

### "GTSAM not available"
```bash
# Install GTSAM (requires Python 3.10 or 3.11)
mamba create -n hloc_stereo python=3.10
mamba activate hloc_stereo
mamba install -c conda-forge gtsam
```

### Windows OpenMP Conflict
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

### Memory Issues
- Use `--sequential_pairs` to reduce matching pairs
- Reduce `max_keypoints` in EXTRACT_CONF
- Process in smaller batches

## Examples

See [run_test.py](run_test.py) for a complete working example.

## Technical Details

### Coordinate Conventions
- **Pose**: `T_world_camera` (transforms camera points to world)
- **Depth**: Meters along camera Z-axis
- **PnP**: `T_rel` transforms points from cam0 to cam1

### PnP Parameters
- Method: P3P with RANSAC
- Reprojection error: 8.0 pixels (for high-res images)
- Min inliers: 6 points
- Inlier ratio threshold: 15%

### BA Parameters
- Optimizer: Levenberg-Marquardt
- Max iterations: 100
- First pose: Fixed with tight prior
- Noise model: Scales inversely with inlier count

## See Also

- [Main README](../../../README.md) - Full usage instructions
- [CLAUDE.md](../../../CLAUDE.md) - Project documentation
- [hloc Documentation](https://github.com/cvg/Hierarchical-Localization)
