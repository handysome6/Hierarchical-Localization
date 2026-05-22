"""Smoke-test the ArUco initializer in isolation.

Skips matching/PnP entirely -- just scans frames, detects markers, and runs
the initializer to confirm it can place frames in a single world frame.
Useful for tuning min_shared_corners and seeing the marker-coverage graph
before paying for a full pipeline run.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from hloc.pipelines.StereoDepth.dataset import scan_frames
from hloc.pipelines.StereoDepth.pose_initializers import ArucoMarkerInitializer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--min_shared_corners", type=int, default=4)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    frames = scan_frames(args.data_dir)
    frame_info = {f.name: f for f in frames}
    frame_names = [f.name for f in frames]

    init = ArucoMarkerInitializer(
        frame_info=frame_info,
        min_shared_corners=args.min_shared_corners,
    )
    poses, unregistered = init.initialize(frame_names, {})

    print(f"\nRegistered: {len(poses)}/{len(frame_names)}")
    print(f"Unregistered: {len(unregistered)}")
    if unregistered:
        preview = unregistered[:10]
        ellipsis = "..." if len(unregistered) > 10 else ""
        print(f"  e.g. {preview}{ellipsis}")

    # Print a short trajectory summary so we can eyeball whether the scale
    # and motion look sensible (no wild jumps).
    if poses:
        translations = np.stack(
            [poses[n][:3, 3] for n in frame_names if n in poses]
        )
        print(f"\nTrajectory bounding box (m):")
        print(f"  min : {translations.min(axis=0)}")
        print(f"  max : {translations.max(axis=0)}")
        print(f"  span: {translations.max(axis=0) - translations.min(axis=0)}")


if __name__ == "__main__":
    main()
