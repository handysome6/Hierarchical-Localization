"""Build 3D-2D correspondences for one directed pair.

Reads the matched keypoints between ``src`` and ``dst`` from hloc's HDF5
files, back-projects the src-side keypoints into the SRC camera frame using
the src depth map, and returns a ``PairCorrespondence`` (or ``None`` if there
is not enough usable data).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from hloc.utils.io import get_keypoints, get_matches

from .types import FrameInfo, PairCorrespondence

logger = logging.getLogger(__name__)


def back_project_with_depth(
    keypoints: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    depth_min: float = 0.1,
    depth_max: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project 2D keypoints to 3D using a depth map.

    Parameters
    ----------
    keypoints : np.ndarray
        Nx2 (u, v) pixel coordinates in the **original full-resolution** image.
    depth_map : np.ndarray
        HxW depth in meters along camera +Z.
    K : np.ndarray
        3x3 intrinsics in the same pixel units as ``keypoints``.
    depth_min, depth_max : float
        Valid-depth gate. Defaults preserve the original pipeline's behavior.

    Returns
    -------
    points_3d : np.ndarray
        Nx3 in camera frame (zeros for invalid rows).
    valid_mask : np.ndarray
        Length-N boolean mask of valid rows.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth_map.shape

    N = len(keypoints)
    points_3d = np.zeros((N, 3), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for i, (u, v) in enumerate(keypoints):
        u_int, v_int = int(round(u)), int(round(v))
        if not (0 <= u_int < W and 0 <= v_int < H):
            continue
        z = depth_map[v_int, u_int]
        if not (depth_min < z < depth_max):
            continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_3d[i] = [x, y, z]
        valid_mask[i] = True

    return points_3d, valid_mask


def build_correspondence(
    src: str,
    dst: str,
    frame_info: Dict[str, FrameInfo],
    features_h5: Path,
    matches_h5: Path,
    depth_min: float = 0.1,
    depth_max: float = 50.0,
    min_matches: int = 10,
    min_depth_valid: int = 10,
) -> Optional[PairCorrespondence]:
    """Assemble the 3D-2D correspondence set for the directed pair (src, dst).

    Returns ``None`` (with a debug log) if there are not enough matches, no
    matches at all, or not enough valid depths on the SRC side. The src depth
    map is loaded on demand.
    """
    src_info = frame_info[src]
    dst_info = frame_info[dst]

    if src_info.depth_path is None:
        logger.debug(f"{src} -> {dst}: src has no depth map")
        return None

    try:
        matches, _ = get_matches(matches_h5, src, dst)
    except ValueError:
        logger.debug(f"{src} -> {dst}: no matches")
        return None

    if len(matches) < min_matches:
        logger.debug(f"{src} -> {dst}: only {len(matches)} matches")
        return None

    kpts_src = get_keypoints(features_h5, src)
    kpts_dst = get_keypoints(features_h5, dst)

    matched_src = kpts_src[matches[:, 0]]
    matched_dst = kpts_dst[matches[:, 1]]

    depth_src = np.load(src_info.depth_path)
    pts_3d, depth_valid = back_project_with_depth(
        matched_src, depth_src, src_info.K, depth_min, depth_max
    )

    if depth_valid.sum() < min_depth_valid:
        logger.debug(f"{src} -> {dst}: only {depth_valid.sum()} depth-valid points")
        return None

    return PairCorrespondence(
        src=src,
        dst=dst,
        pts_3d=pts_3d[depth_valid],
        pts_2d=matched_dst[depth_valid],
        K_src=src_info.K,
        K_dst=dst_info.K,
        match_idx=matches[depth_valid],
    )
