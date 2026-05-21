"""Shared data types for the StereoDepth pipeline.

These dataclasses are the contracts between the pipeline stages. Every field's
**structural meaning** (frame of reference, units, conventions) is documented so
that swapping any single module remains safe.

Conventions (locked, see /home/andy/.claude/plans/choose-yourself-keen-jellyfish.md):

- Image name: bare filename string, e.g. "1700000000.jpg".
- Intrinsics ``K``: 3x3, pixel units of the **original full-resolution** image
  (hloc rescales keypoints back to this).
- Depth map: HxW float32, meters along camera +Z; values <=0 or NaN are invalid.
- ``T_world_cam``: 4x4 homogeneous transform; ``p_world = T_world_cam @ p_cam``.
  Used for every **global** pose.
- ``T_rel`` (src -> dst): 4x4 homogeneous transform; ``p_dst = T_rel @ p_src``.
  Used for every **pairwise** transform (both the per-pair estimator output and
  the BA edge).
- Pose-graph composition: ``T_world_dst = T_world_src @ inv(T_rel)``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameInfo:
    """Per-frame metadata loaded from disk."""

    name: str
    """Canonical image name used as the dictionary key everywhere."""

    image_path: Path
    """Path to the rectified left image."""

    K: np.ndarray
    """3x3 camera intrinsics in pixel units of the original full-res image."""

    baseline: float
    """Stereo baseline in meters."""

    depth_path: Optional[Path]
    """HxW float32 depth map in meters, along camera +Z. None if depth-less."""

    cloud_path: Optional[Path]
    """Per-frame point cloud in the CAMERA frame. None if missing."""


@dataclass
class PairCorrespondence:
    """3D-2D correspondences for a single directed pair (src -> dst).

    Produced by ``correspondence.build_correspondence`` from features, matches
    and the src-frame depth map. Consumed by any ``PairPoseEstimator``.
    """

    src: str
    dst: str

    pts_3d: np.ndarray
    """Nx3, in SRC camera frame, meters. Only depth-valid points are kept."""

    pts_2d: np.ndarray
    """Nx2, in DST image, pixel coordinates at original full-res."""

    K_src: np.ndarray
    """3x3 intrinsics of the SRC camera (used for back-projection)."""

    K_dst: np.ndarray
    """3x3 intrinsics of the DST camera (used for the PnP projection model)."""

    match_idx: np.ndarray
    """Nx2 of (kpt_src_idx, kpt_dst_idx) into the original keypoint arrays.

    Kept so downstream stages can trace an inlier back to its raw keypoint.
    """


@dataclass
class RelativePose:
    """A directed pairwise pose estimate src -> dst.

    Produced by any ``PairPoseEstimator``. The ``T_rel`` convention is
    ``p_dst = T_rel @ p_src`` (the same as what OpenCV's PnP produces).
    """

    src: str
    dst: str

    T_rel: np.ndarray
    """4x4 homogeneous transform, ``p_dst = T_rel @ p_src``."""

    inliers: np.ndarray
    """Indices into the input ``PairCorrespondence`` arrays."""

    num_inliers: int
    """Convenience cache of ``len(inliers)``."""

    score: float
    """Estimator-defined confidence (e.g. inlier ratio in [0, 1])."""

    num_matches: int
    """Total matches considered before the estimator ran. For logging only."""


# A registered global pose set. Keys are image names, values are 4x4
# ``T_world_cam`` matrices. Frames absent from the dict are unregistered.
GlobalPoses = Dict[str, np.ndarray]


@dataclass
class PoseGraphEdge:
    """One edge in the pose graph consumed by a ``PoseGraphOptimizer``.

    The optimizer is agnostic to where the edge came from (PnP, IMU,
    wheel odometry, learned regressor, ...). All it needs is the directed
    relative transform and a scalar weight from which to derive a noise model.
    """

    src: str
    dst: str

    T_rel: np.ndarray
    """4x4 homogeneous transform, ``p_dst = T_rel @ p_src``."""

    weight: float
    """Edge confidence. The current GTSAM optimizer derives an isotropic
    sigma as ``base_sigma * max(0.5, 100 / weight)``; pass num_inliers from
    PnP, or any other monotonic confidence measure.
    """


# Type alias for the keying scheme used by ``relative_poses`` dictionaries:
# ``(src, dst) -> RelativePose``. The keys are **directed** because some
# initializers care about direction (depth-PnP is asymmetric).
RelativePoseMap = Dict[Tuple[str, str], RelativePose]
