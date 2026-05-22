"""Validate ArUco detection on the publichouse2 dataset.

Walks ``data_dir/<timestamp>/rect_left.jpg``, runs the ArUco detector with
``detectInvertedMarker=True`` (so it picks up both normal black-on-white and
color-reversed white-on-black markers), and prints per-frame counts.

Optionally writes annotated previews of the first few frames to ``out_dir``
so we can eyeball detector quality and corner ordering before wiring it into
the pose initializer.

Usage:
    python -m hloc.pipelines.StereoDepth.tools.aruco_detect_validate \\
        --data_dir /home/andy/Downloads/publichouse2 \\
        --out_dir /tmp/aruco_vis \\
        --num_preview 4
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Dictionaries to try, ordered by how commonly they show up. We return as soon
# as one dictionary produces detections so we don't double-count markers that
# happen to also decode under a smaller dictionary.
DICT_CANDIDATES = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
]


def make_detector(dict_id: int) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Critical for this dataset: marker interiors may be color-reversed.
    params.detectInvertedMarker = True
    # Refine corner locations to subpixel — matters because we will sample
    # depth/XYZ at the corner pixels later.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_best_dict(img_gray: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
    """Try each candidate dict; return the first one with >0 detections.

    Returns (dict_name, corners, ids). corners is a list-like of (1, 4, 2)
    arrays as returned by OpenCV; ids is an (N, 1) int array. If no dict
    yields a hit, returns ("", [], None).
    """
    for name, dict_id in DICT_CANDIDATES:
        detector = make_detector(dict_id)
        corners, ids, _ = detector.detectMarkers(img_gray)
        if ids is not None and len(ids) > 0:
            return name, corners, ids
    return "", [], None


def annotate(img_bgr: np.ndarray, corners, ids) -> np.ndarray:
    out = img_bgr.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Optional: write annotated previews here",
    )
    ap.add_argument(
        "--num_preview",
        type=int,
        default=4,
        help="How many frames to save annotated previews for",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Process only this many frames (0=all)"
    )
    args = ap.parse_args()

    frame_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    if args.limit:
        frame_dirs = frame_dirs[: args.limit]

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    dict_hits: Counter = Counter()
    id_hits: Counter = Counter()
    detected_frames = 0
    no_marker_frames: List[str] = []
    preview_written = 0

    for fdir in frame_dirs:
        img_path = fdir / "rect_left.jpg"
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        dict_name, corners, ids = detect_best_dict(img_gray)
        if ids is None or len(ids) == 0:
            no_marker_frames.append(fdir.name)
            print(f"{fdir.name}: no markers")
            continue

        detected_frames += 1
        dict_hits[dict_name] += 1
        for mid in ids.flatten().tolist():
            id_hits[int(mid)] += 1

        print(
            f"{fdir.name}: dict={dict_name} "
            f"n={len(ids)} ids={ids.flatten().tolist()}"
        )

        if args.out_dir is not None and preview_written < args.num_preview:
            vis = annotate(img_bgr, corners, ids)
            # Downscale to keep previews manageable.
            scale = 1024 / max(vis.shape[:2])
            if scale < 1.0:
                vis = cv2.resize(
                    vis,
                    (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            out_path = args.out_dir / f"{fdir.name}_aruco.jpg"
            cv2.imwrite(str(out_path), vis)
            preview_written += 1

    print("\n=== Summary ===")
    print(f"frames processed     : {len(frame_dirs)}")
    print(f"frames with markers  : {detected_frames}")
    print(f"frames without       : {len(no_marker_frames)}")
    print(f"dict distribution    : {dict(dict_hits)}")
    print(f"marker ID histogram  : {dict(sorted(id_hits.items()))}")


if __name__ == "__main__":
    main()
