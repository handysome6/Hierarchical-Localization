"""StereoDepth pipeline: hloc matching + depth-based PnP + GTSAM pose-graph BA.

This file is the thin orchestrator. Every stage now lives in its own module
under ``hloc/pipelines/StereoDepth/``:

- ``dataset``           — scan disk -> ``list[FrameInfo]``
- ``pair_selection``    — pair-list policies (exhaustive / sequential)
- ``matching``          — hloc feature extraction + matching
- ``correspondence``    — per-pair 3D-2D correspondence assembly
- ``pose_estimators``   — per-pair pose (default: ``DepthPnPEstimator``)
- ``pose_initializers`` — global init (incremental / sequential / from JSON)
- ``optimizers``        — pose-graph optimization (default: ``GtsamPoseGraphOptimizer``)
- ``fusion``            — point-cloud concatenation
- ``io``                — pose I/O

The CLI and ``run_pipeline(...)`` signature are preserved so existing scripts
(notably ``run_test.py``) keep working.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .correspondence import build_correspondence
from .dataset import scan_frames
from .fusion import merge_point_clouds
from .io import save_pose_results, save_poses_txt
from .matching import run_matching
from .optimizers import GTSAM_AVAILABLE, GtsamPoseGraphOptimizer
from .pair_selection import generate_pairs
from .pose_estimators import DepthPnPEstimator
from .pose_initializers import (
    IncrementalInitializer,
    JsonInitializer,
    SequentialInitializer,
)
from .types import (
    FrameInfo,
    GlobalPoses,
    PoseGraphEdge,
    RelativePose,
    RelativePoseMap,
)

logger = logging.getLogger(__name__)

# Re-exported for backward compatibility with ``run_test.py``:
# ``from pipeline import run_pipeline, GTSAM_AVAILABLE``.
__all__ = ["run_pipeline", "GTSAM_AVAILABLE", "main"]


def _estimate_all_relative_poses(
    pairs: List,
    frame_info: Dict[str, FrameInfo],
    features_h5: Path,
    matches_h5: Path,
    estimator: DepthPnPEstimator,
    try_both_directions: bool = True,
) -> RelativePoseMap:
    """Run the per-pair estimator across every pair (and optionally each
    direction). Returns a directed ``(src, dst) -> RelativePose`` map.
    """
    relative_poses: RelativePoseMap = {}
    for img0, img1 in tqdm(pairs, desc="Per-pair PnP"):
        directions = (
            [(img0, img1), (img1, img0)] if try_both_directions else [(img0, img1)]
        )
        for src, dst in directions:
            corr = build_correspondence(src, dst, frame_info, features_h5, matches_h5)
            if corr is None:
                continue
            rel = estimator.estimate(corr)
            if rel is None:
                continue
            relative_poses[(src, dst)] = rel
    logger.info(f"Estimated {len(relative_poses)} directed relative poses")
    return relative_poses


def _build_ba_edges(
    relative_poses: RelativePoseMap,
    init_poses: GlobalPoses,
    min_inliers: int,
    consistency_check: bool = True,
    consistency_translation_min: float = 0.5,
) -> List[PoseGraphEdge]:
    """Construct deduplicated, filtered pose-graph edges for BA.

    Steps:

    1. Collapse the directed relative-pose map to one entry per **undirected**
       pair, keeping the direction with more inliers.
    2. Drop edges below ``min_inliers``.
    3. (Optional) Drop edges whose translation magnitude disagrees with the
       initial poses by more than 50% (or 0.5m, whichever is larger). This
       filter — present only on the incremental path in the prior code — is
       generalized here because mismatched edges hurt BA regardless of init.
    """
    # Step 1: dedupe by canonical (src, dst) pair.
    best_by_canonical: Dict[tuple, RelativePose] = {}
    for (src, dst), rel in relative_poses.items():
        if src not in init_poses or dst not in init_poses:
            continue
        key = tuple(sorted((src, dst)))
        cur = best_by_canonical.get(key)
        if cur is None or rel.num_inliers > cur.num_inliers:
            best_by_canonical[key] = rel

    edges: List[PoseGraphEdge] = []
    n_inlier_filtered = 0
    n_consistency_filtered = 0
    for rel in best_by_canonical.values():
        if rel.num_inliers < min_inliers:
            n_inlier_filtered += 1
            continue

        if consistency_check:
            T_src = init_poses[rel.src]
            T_dst = init_poses[rel.dst]
            # Expected T_rel from poses: T_world_dst = T_world_src @ inv(T_rel)
            # => inv(T_rel) = inv(T_src) @ T_dst => T_rel = inv(inv(T_src) @ T_dst)
            T_rel_expected = np.linalg.inv(np.linalg.inv(T_src) @ T_dst)
            t_measured = float(np.linalg.norm(rel.T_rel[:3, 3]))
            t_expected = float(np.linalg.norm(T_rel_expected[:3, 3]))
            thresh = max(0.5 * max(t_measured, t_expected), consistency_translation_min)
            if abs(t_measured - t_expected) > thresh:
                n_consistency_filtered += 1
                logger.debug(
                    f"Drop BA edge {rel.src} -> {rel.dst}: "
                    f"|t| measured={t_measured:.3f}m, expected={t_expected:.3f}m"
                )
                continue

        edges.append(
            PoseGraphEdge(
                src=rel.src, dst=rel.dst, T_rel=rel.T_rel, weight=float(rel.num_inliers)
            )
        )

    logger.info(
        f"BA edges: {len(edges)} kept from {len(best_by_canonical)} unique pairs "
        f"({n_inlier_filtered} below inlier threshold, "
        f"{n_consistency_filtered} failed consistency)"
    )
    return edges


def _pose_results_from_relpose_map(rels: RelativePoseMap) -> List[dict]:
    """Convert the directed relative-pose map to the legacy ``pose_results.json``
    record format (one entry per directed pair)."""
    return [
        {
            "frame0": rel.src,
            "frame1": rel.dst,
            "success": True,
            "num_matches": rel.num_matches,
            "num_inliers": rel.num_inliers,
            "translation": float(np.linalg.norm(rel.T_rel[:3, 3])),
        }
        for rel in rels.values()
    ]


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    visualize: bool = False,
    concatenate_pcd: bool = True,
    voxel_size: float = 0.01,
    run_ba: bool = False,
    exhaustive_pairs: bool = True,
    max_exhaustive_frames: int = 50,
    incremental_init: bool = True,
    min_init_inliers: int = 15,
    min_ba_inliers: int = 30,
    extrinsics_json: Optional[Path] = None,
) -> dict:
    """Run the stereo depth pipeline end-to-end.

    Signature preserved verbatim for backward compatibility with ``run_test.py``
    and the CLI shim. See module docstring for the per-stage modules invoked.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: dataset scan.
    frames = scan_frames(Path(data_dir))
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames, found {len(frames)}")
    frame_info = {f.name: f for f in frames}
    frame_names = [f.name for f in frames]

    # Stage 2: pair list.
    mode = "exhaustive" if exhaustive_pairs else "sequential"
    pairs = generate_pairs(frames, mode=mode, max_exhaustive=max_exhaustive_frames)
    use_exhaustive = exhaustive_pairs and len(frames) <= max_exhaustive_frames

    # Stage 3: feature extraction + matching.
    features_h5, matches_h5 = run_matching(frames, pairs, output_dir)

    # Stage 4: per-pair relative pose estimation. We always try both directions
    # because depth-based PnP is asymmetric (the src side needs valid depth).
    estimator = DepthPnPEstimator()
    relative_poses = _estimate_all_relative_poses(
        pairs, frame_info, features_h5, matches_h5, estimator, try_both_directions=True
    )

    # Stage 5: global pose initialization. The choice of initializer is the
    # main per-image-init swap-point the user asked about; new initializers
    # only need to implement the ``GlobalPoseInitializer`` protocol.
    if extrinsics_json is not None:
        initializer = JsonInitializer(Path(extrinsics_json))
    elif use_exhaustive and incremental_init:
        initializer = IncrementalInitializer(min_inliers=min_init_inliers)
    else:
        initializer = SequentialInitializer()

    poses, unregistered = initializer.initialize(frame_names, relative_poses)
    if unregistered:
        logger.warning(f"Unregistered frames: {unregistered}")

    pose_results = _pose_results_from_relpose_map(relative_poses)

    # Stage 6: optional pose-graph BA. The optimizer is generic over edges,
    # so any source of pairwise transforms (PnP today, IMU/odometry/etc.
    # tomorrow) can feed it.
    if run_ba and GTSAM_AVAILABLE and relative_poses:
        edges = _build_ba_edges(
            relative_poses,
            poses,
            min_inliers=max(min_init_inliers, min_ba_inliers),
        )
        if edges:
            optimizer = GtsamPoseGraphOptimizer()
            poses = optimizer.optimize(poses, edges)
        else:
            logger.warning("No BA edges survived filtering; skipping optimization")
    elif run_ba and not GTSAM_AVAILABLE:
        logger.warning("--ba requested but GTSAM unavailable; skipping")

    # Stage 7: pose I/O.
    save_poses_txt(poses, output_dir / "poses.txt")
    save_pose_results(pose_results, output_dir / "pose_results.json")

    # Stage 8: optional point-cloud fusion.
    if concatenate_pcd:
        merge_point_clouds(frames, poses, output_dir / "combined.ply", voxel_size)

    return {
        "num_frames": len(frames),
        "num_successful": sum(1 for r in pose_results if r.get("success", False)),
        "poses": poses,
        "pose_results": pose_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stereo Depth Pipeline: hloc matching + depth-based PnP"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing timestamp folders with stereo data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: data_dir/hloc_output)",
    )
    parser.add_argument(
        "--no_pcd", action="store_true", help="Skip point cloud concatenation"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for point cloud downsampling (default: 0.01m)",
    )
    parser.add_argument(
        "--ba", action="store_true", help="Run Bundle Adjustment using GTSAM"
    )
    parser.add_argument(
        "--sequential_pairs",
        action="store_true",
        help="Use sequential pairing only (default: exhaustive for small datasets)",
    )
    parser.add_argument(
        "--max_exhaustive",
        type=int,
        default=50,
        help="Max frames for exhaustive pairing (default: 50)",
    )
    parser.add_argument(
        "--no_incremental_init",
        action="store_true",
        help="Disable incremental initialization (use sequential initialization)",
    )
    parser.add_argument(
        "--min_init_inliers",
        type=int,
        default=15,
        help="Minimum inlier count for incremental initialization (default: 15)",
    )
    parser.add_argument(
        "--min_ba_inliers",
        type=int,
        default=30,
        help=(
            "Minimum inlier count for BA edges "
            "(default: 30, higher = more conservative)"
        ),
    )
    parser.add_argument(
        "--extrinsics_json",
        type=Path,
        default=None,
        help="JSON file with pre-computed camera poses (skips pose estimation). "
        'Format: {"<timestamp>.jpg": [[r11,r12,r13,t1], [r21,r22,r23,t2], '
        "[r31,r32,r33,t3]], ...}",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.output_dir is None:
        args.output_dir = args.data_dir / "hloc_output"

    results = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        concatenate_pcd=not args.no_pcd,
        voxel_size=args.voxel_size,
        run_ba=args.ba,
        exhaustive_pairs=not args.sequential_pairs,
        max_exhaustive_frames=args.max_exhaustive,
        incremental_init=not args.no_incremental_init,
        min_init_inliers=args.min_init_inliers,
        min_ba_inliers=args.min_ba_inliers,
        extrinsics_json=args.extrinsics_json,
    )

    logger.info(
        f"Pipeline complete: {results['num_successful']}/"
        f"{results['num_frames'] - 1} successful pose estimates"
    )


if __name__ == "__main__":
    main()
