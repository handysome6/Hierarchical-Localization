"""Pose-graph optimizers.

Each optimizer consumes ``(GlobalPoses, list[PoseGraphEdge])`` and returns a
refined ``GlobalPoses``. Optimizers know nothing about depth, PnP, cameras,
or hloc — they operate purely on the pose graph.
"""

from .base import PoseGraphOptimizer
from .gtsam_pose_graph import GTSAM_AVAILABLE, GtsamPoseGraphOptimizer

__all__ = [
    "PoseGraphOptimizer",
    "GtsamPoseGraphOptimizer",
    "GTSAM_AVAILABLE",
]
