"""hloc feature extraction + matching wrapper.

Owns the disk staging (copying each frame's rectified image into a flat
``output_dir/images/`` directory under its canonical name) and the
``pairs.txt`` file required by ``hloc.match_features``.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from hloc import extract_features, match_features

from .types import FrameInfo

logger = logging.getLogger(__name__)


# Default SuperPoint extraction config (high-res, 4096 keypoints).
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

# Default LightGlue matching config (paired with SuperPoint).
MATCH_CONF = {
    "output": "matches-lightglue",
    "model": {
        "name": "lightglue",
        "features": "superpoint",
    },
}


def _stage_images(frames: List[FrameInfo], images_dir: Path) -> None:
    """Copy each frame's image into ``images_dir`` under its canonical name."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        dst = images_dir / f.name
        if not dst.exists():
            shutil.copy(f.image_path, dst)


def _write_pairs_file(pairs: List[Tuple[str, str]], pairs_path: Path) -> None:
    with open(pairs_path, "w") as fh:
        for a, b in pairs:
            fh.write(f"{a} {b}\n")


def run_matching(
    frames: List[FrameInfo],
    pairs: List[Tuple[str, str]],
    output_dir: Path,
    extract_conf: dict = EXTRACT_CONF,
    match_conf: dict = MATCH_CONF,
) -> Tuple[Path, Path]:
    """Run hloc extraction + matching.

    Parameters
    ----------
    frames : list[FrameInfo]
        Frames whose images will be staged and processed.
    pairs : list[tuple[str, str]]
        Ordered pair list (image names). Direction is preserved.
    output_dir : Path
        Where ``images/``, ``features.h5``, ``pairs.txt``, ``matches.h5``
        are written.

    Returns
    -------
    features_h5 : Path
    matches_h5 : Path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    _stage_images(frames, images_dir)
    logger.info(f"Staged {len(frames)} images in {images_dir}")

    features_path = output_dir / "features.h5"
    logger.info(f"Extracting features with {extract_conf['model']['name']}...")
    extract_features.main(
        conf=extract_conf,
        image_dir=images_dir,
        export_dir=output_dir,
        feature_path=features_path,
    )

    pairs_path = output_dir / "pairs.txt"
    _write_pairs_file(pairs, pairs_path)
    logger.info(f"Wrote {len(pairs)} pairs to {pairs_path}")

    matches_path = output_dir / "matches.h5"
    logger.info(f"Matching features with {match_conf['model']['name']}...")
    match_features.main(
        conf=match_conf,
        pairs=pairs_path,
        features=features_path,
        export_dir=output_dir,
        matches=matches_path,
    )

    return features_path, matches_path
