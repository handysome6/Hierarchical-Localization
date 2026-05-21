"""Global pose initializer that loads poses from a JSON file.

Useful when an external system (a prior SfM run, a ground-truth tracker, IMU
preintegration, etc.) already provides per-frame poses, and the pipeline is
only used to refine them with BA or to fuse point clouds.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..types import GlobalPoses, RelativePoseMap

logger = logging.getLogger(__name__)


class JsonInitializer:
    """Load poses from a JSON file.

    Expected JSON shape::

        {
            "<image_name>": [[r11, r12, r13, t1],
                             [r21, r22, r23, t2],
                             [r31, r32, r33, t3]],
            ...
        }

    Each value is a 3x4 ``[R | t]`` block. It is treated as a
    ``T_world_cam`` matrix and padded to 4x4.
    """

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self._cache: GlobalPoses = self._load()

    def _load(self) -> GlobalPoses:
        with open(self.json_path, "r") as f:
            data = json.load(f)
        poses: GlobalPoses = {}
        for name, mat_3x4 in data.items():
            T = np.eye(4)
            T[:3, :] = np.array(mat_3x4)
            poses[name] = T
        logger.info(f"Loaded {len(poses)} poses from {self.json_path}")
        return poses

    def initialize(
        self,
        frame_names: List[str],
        relative_poses: RelativePoseMap,
    ) -> Tuple[GlobalPoses, List[str]]:
        # relative_poses is intentionally ignored; this initializer is a
        # pure data loader. The orchestrator may still hand the same map to
        # an optimizer afterwards.
        poses: GlobalPoses = {}
        unregistered: List[str] = []
        for name in frame_names:
            if name in self._cache:
                poses[name] = self._cache[name]
            else:
                unregistered.append(name)
                logger.warning(f"No pose for {name} in {self.json_path}")
        logger.info(f"JSON init: {len(poses)}/{len(frame_names)} placed")
        return poses, unregistered
