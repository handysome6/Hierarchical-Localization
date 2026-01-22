#!/usr/bin/env python
"""Test script with OpenMP fix."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from pathlib import Path
from pipeline import run_pipeline, GTSAM_AVAILABLE

print(f"GTSAM_AVAILABLE: {GTSAM_AVAILABLE}")

results = run_pipeline(
    data_dir=Path(r"C:\workspace\stereo_vo_reconstruction\0119_office2"),
    output_dir=Path(r"C:\workspace\stereo_vo_reconstruction\0119_office2\hloc_ba_output"),
    concatenate_pcd=True,
    voxel_size=0.01,
    run_ba=True,
    exhaustive_pairs=True,  # Use all pairs for dense BA constraints
)

print(f"Pipeline complete: {results['num_successful']}/{results['num_frames']-1} successful")


# python -m hloc.pipelines.StereoDepth.pipeline --data_dir "C:\workspace\stereo_vo_reconstruction\climb_indoor" --output_dir "C:\workspace\stereo_vo_reconstruction\climb_indoor\hloc_ba_output" --ba --voxel_size 0.01