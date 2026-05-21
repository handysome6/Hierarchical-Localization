"""Dataset scanning: disk layout -> list[FrameInfo].

Expected layout::

    data_dir/
        {timestamp1}/
            rect_left.jpg
            depth_meter.npy
            cloud.ply
            K.txt
        {timestamp2}/
            ...

``K.txt`` format::

    fx 0 cx 0 fy cy 0 0 1
    baseline_in_meters
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .types import FrameInfo

logger = logging.getLogger(__name__)


def load_camera_intrinsics(k_txt_path: Path) -> Tuple[np.ndarray, float]:
    """Parse a ``K.txt`` file.

    Returns
    -------
    K : np.ndarray
        3x3 intrinsic matrix in original-image pixel units.
    baseline : float
        Stereo baseline in meters.
    """
    with open(k_txt_path, "r") as f:
        lines = f.readlines()

    k_values = list(map(float, lines[0].strip().split()))
    K = np.array(k_values).reshape(3, 3)
    baseline = float(lines[1].strip())
    return K, baseline


def scan_frames(data_dir: Path) -> List[FrameInfo]:
    """Scan ``data_dir`` for timestamp folders and load per-frame metadata.

    Folders whose names are not all digits are ignored. Folders missing
    ``rect_left.jpg`` are skipped with a warning. Frames are returned in
    timestamp order.
    """
    timestamp_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()
    )

    frames: List[FrameInfo] = []
    for ts_dir in timestamp_dirs:
        left_img = ts_dir / "rect_left.jpg"
        if not left_img.exists():
            logger.warning(f"Missing rect_left.jpg in {ts_dir}")
            continue

        K, baseline = load_camera_intrinsics(ts_dir / "K.txt")

        depth_path = ts_dir / "depth_meter.npy"
        cloud_path = ts_dir / "cloud.ply"

        frames.append(
            FrameInfo(
                name=f"{ts_dir.name}.jpg",
                image_path=left_img,
                K=K,
                baseline=baseline,
                depth_path=depth_path if depth_path.exists() else None,
                cloud_path=cloud_path if cloud_path.exists() else None,
            )
        )

    logger.info(f"Scanned {len(frames)} frames from {data_dir}")
    return frames
