"""Pose I/O for the StereoDepth pipeline.

We keep our own ``poses.txt`` reader/writer (instead of reusing
``hloc.utils.io.write_poses``, which expects ``pycolmap.Rigid3d``) because the
on-disk format is the simple 4x4 matrix block that downstream tools in this
repo already read.
"""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from .types import GlobalPoses

logger = logging.getLogger(__name__)


def save_poses_txt(poses: GlobalPoses, path: Path) -> None:
    """Write poses as ``# <name>`` headers followed by four 4-value lines."""
    with open(path, "w") as f:
        for name, T in poses.items():
            f.write(f"# {name}\n")
            for row in T:
                f.write(" ".join(f"{v:.8f}" for v in row) + "\n")
    logger.info(f"Saved {len(poses)} poses to {path}")


def save_pose_results(results: List[dict], path: Path) -> None:
    """Dump per-pair pose statistics as JSON for inspection."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_poses_json(path: Path) -> GlobalPoses:
    """Load poses from a JSON file of ``{name: [[r|t]_3x4]}`` entries.

    Each value is padded to a 4x4 ``T_world_cam`` matrix.
    """
    with open(path, "r") as f:
        data = json.load(f)
    poses: GlobalPoses = {}
    for name, mat_3x4 in data.items():
        T = np.eye(4)
        T[:3, :] = np.array(mat_3x4)
        poses[name] = T
    logger.info(f"Loaded {len(poses)} poses from {path}")
    return poses
