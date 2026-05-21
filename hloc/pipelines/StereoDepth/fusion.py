"""Per-frame point-cloud fusion under estimated poses."""

import logging
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .types import FrameInfo, GlobalPoses

logger = logging.getLogger(__name__)


def merge_point_clouds(
    frames: Iterable[FrameInfo],
    poses: GlobalPoses,
    output_path: Path,
    voxel_size: float = 0.01,
) -> int:
    """Concatenate per-frame camera-frame point clouds into one world cloud.

    Each frame's ``cloud_path`` is read, transformed by its ``T_world_cam``
    pose, and accumulated. The result is optionally voxel-downsampled and
    written to ``output_path``.

    Returns the number of points written, or 0 if nothing was emitted.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("open3d not installed; skipping point cloud fusion")
        return 0

    combined = o3d.geometry.PointCloud()

    for f in tqdm(list(frames), desc="Fusing point clouds"):
        if f.cloud_path is None or not f.cloud_path.exists():
            logger.warning(f"Missing point cloud for {f.name}")
            continue
        if f.name not in poses:
            logger.warning(f"No pose for {f.name}")
            continue
        pcd = o3d.io.read_point_cloud(str(f.cloud_path))
        pcd.transform(poses[f.name])
        combined += pcd

    n = len(combined.points)
    if n == 0:
        logger.error("Nothing to fuse")
        return 0

    logger.info(f"Combined cloud: {n} points before downsampling")
    if voxel_size > 0:
        combined = combined.voxel_down_sample(voxel_size)
        logger.info(f"After voxel({voxel_size}m): {len(combined.points)} points")

    o3d.io.write_point_cloud(str(output_path), combined)
    logger.info(f"Wrote {len(combined.points)} points to {output_path}")
    return len(combined.points)
