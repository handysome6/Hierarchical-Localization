"""GTSAM pose-graph optimizer using ``BetweenFactorPose3``.

The optimizer is generic: it accepts any directed ``PoseGraphEdge`` regardless
of source (PnP, IMU preintegration, wheel odometry, learned regressor). Edge
filtering (inlier thresholds, geometric consistency) is the caller's concern.
"""

import logging
import traceback
from typing import List, Optional

import numpy as np

from ..types import GlobalPoses, PoseGraphEdge

logger = logging.getLogger(__name__)

try:
    import gtsam
    from gtsam import symbol_shorthand

    X = symbol_shorthand.X  # Pose
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    logger.warning("GTSAM not available, pose-graph BA will be a no-op")


def _pose_to_gtsam(T: np.ndarray) -> "gtsam.Pose3":
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))


def _gtsam_to_pose(pose3: "gtsam.Pose3") -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = pose3.rotation().matrix()
    T[:3, 3] = pose3.translation()
    return T


class GtsamPoseGraphOptimizer:
    """Levenberg-Marquardt pose-graph optimization.

    Parameters
    ----------
    max_iterations, rel_error_tol, abs_error_tol : LM stop criteria.
    fix_anchor : if True, add a tight prior on the anchor frame.
    base_rotation_sigma, base_translation_sigma : isotropic noise prior, in
        radians and meters respectively. Per-edge noise is scaled by
        ``max(0.5, 100 / edge.weight)`` so higher-confidence edges (e.g. more
        PnP inliers) get tighter constraints.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        rel_error_tol: float = 1e-5,
        abs_error_tol: float = 1e-5,
        fix_anchor: bool = True,
        base_rotation_sigma: float = 0.05,
        base_translation_sigma: float = 0.05,
        anchor_sigma: float = 1e-3,
    ):
        self.max_iterations = max_iterations
        self.rel_error_tol = rel_error_tol
        self.abs_error_tol = abs_error_tol
        self.fix_anchor = fix_anchor
        self.base_rotation_sigma = base_rotation_sigma
        self.base_translation_sigma = base_translation_sigma
        self.anchor_sigma = anchor_sigma

    def optimize(
        self,
        initial_poses: GlobalPoses,
        edges: List[PoseGraphEdge],
        anchor: Optional[str] = None,
    ) -> GlobalPoses:
        if not GTSAM_AVAILABLE:
            logger.warning("GTSAM unavailable; returning initial poses unchanged")
            return initial_poses

        if len(edges) < 2:
            logger.warning(f"Only {len(edges)} edges; skipping BA")
            return initial_poses

        # Determine which frames have at least one factor edge. GTSAM will
        # error on variables that appear in `initial` but in zero factors, so
        # isolated frames are skipped and their initial pose passed through.
        frame_names = list(initial_poses.keys())
        frame_to_idx = {name: i for i, name in enumerate(frame_names)}

        constrained = set()
        for e in edges:
            constrained.add(e.src)
            constrained.add(e.dst)

        isolated = [n for n in frame_names if n not in constrained]
        if isolated:
            logger.warning(
                f"Skipping {len(isolated)} unconstrained frame(s) in BA: " f"{isolated}"
            )

        # Pick an anchor that actually participates in the factor graph.
        if anchor is None:
            anchor = frame_names[0]
        if anchor not in constrained:
            if not constrained:
                logger.warning("No constrained frames; skipping BA")
                return initial_poses
            new_anchor = next(n for n in frame_names if n in constrained)
            logger.warning(
                f"Anchor {anchor!r} is isolated; using {new_anchor!r} instead"
            )
            anchor = new_anchor

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        for name in frame_names:
            if name in constrained:
                initial.insert(
                    X(frame_to_idx[name]), _pose_to_gtsam(initial_poses[name])
                )

        if self.fix_anchor:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.full(6, self.anchor_sigma)
            )
            graph.add(
                gtsam.PriorFactorPose3(
                    X(frame_to_idx[anchor]),
                    _pose_to_gtsam(initial_poses[anchor]),
                    prior_noise,
                )
            )

        # Add BetweenFactorPose3 for every edge. With the locked convention
        # `p_dst = T_rel @ p_src`, no inversion is needed here:
        # BetweenFactor expects exactly that "src to dst" transform.
        base_sigmas = np.array(
            [
                self.base_rotation_sigma,
                self.base_rotation_sigma,
                self.base_rotation_sigma,
                self.base_translation_sigma,
                self.base_translation_sigma,
                self.base_translation_sigma,
            ]
        )
        for e in edges:
            scale = max(0.5, 100.0 / max(e.weight, 1.0))
            noise = gtsam.noiseModel.Diagonal.Sigmas(base_sigmas * scale)
            graph.add(
                gtsam.BetweenFactorPose3(
                    X(frame_to_idx[e.src]),
                    X(frame_to_idx[e.dst]),
                    _pose_to_gtsam(e.T_rel),
                    noise,
                )
            )

        logger.info(f"Pose-graph BA: {len(edges)} edges, anchor={anchor}")

        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(self.max_iterations)
            params.setRelativeErrorTol(self.rel_error_tol)
            params.setAbsoluteErrorTol(self.abs_error_tol)

            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = optimizer.optimize()

            optimized: GlobalPoses = {}
            for name in frame_names:
                if name in constrained:
                    optimized[name] = _gtsam_to_pose(
                        result.atPose3(X(frame_to_idx[name]))
                    )
                else:
                    optimized[name] = initial_poses[name]

            logger.info(
                f"BA error {graph.error(initial):.4f} -> "
                f"{graph.error(result):.4f} "
                f"({optimizer.iterations()} iters)"
            )
            return optimized
        except Exception as ex:
            logger.error(f"BA failed: {ex}")
            traceback.print_exc()
            return initial_poses
