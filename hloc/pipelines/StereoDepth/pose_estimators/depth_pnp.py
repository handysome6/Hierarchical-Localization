"""PnP + RANSAC pose estimator on depth-back-projected correspondences."""

import logging
from typing import Optional

import cv2
import numpy as np

from ..types import PairCorrespondence, RelativePose

logger = logging.getLogger(__name__)


class DepthPnPEstimator:
    """OpenCV P3P + RANSAC, parameterized for high-res stereo rigs.

    The constructor parameters are the only knobs an alternate estimator might
    expose; defaults match the previous monolithic ``estimate_pose_pnp``.
    """

    def __init__(
        self,
        iterations: int = 2000,
        reprojection_error: float = 8.0,
        confidence: float = 0.99,
        max_translation_m: float = 10.0,
        min_inlier_ratio: float = 0.15,
        min_inliers: int = 6,
        flags: int = cv2.SOLVEPNP_P3P,
    ):
        self.iterations = iterations
        self.reprojection_error = reprojection_error
        self.confidence = confidence
        self.max_translation_m = max_translation_m
        self.min_inlier_ratio = min_inlier_ratio
        self.min_inliers = min_inliers
        self.flags = flags

    def estimate(self, corr: PairCorrespondence) -> Optional[RelativePose]:
        pts_3d = np.ascontiguousarray(corr.pts_3d, dtype=np.float64)
        pts_2d = np.ascontiguousarray(corr.pts_2d, dtype=np.float64)
        K = np.ascontiguousarray(corr.K_dst, dtype=np.float64)
        dist = np.zeros(4, dtype=np.float64)

        if len(pts_3d) < self.min_inliers:
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            K,
            dist,
            iterationsCount=self.iterations,
            reprojectionError=self.reprojection_error,
            confidence=self.confidence,
            flags=self.flags,
        )

        if not success or inliers is None or len(inliers) < self.min_inliers:
            return None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        # Reject implausibly large per-pair translations.
        if np.linalg.norm(t) > self.max_translation_m:
            logger.warning(
                f"{corr.src} -> {corr.dst}: large translation "
                f"{np.linalg.norm(t):.2f}m, rejecting"
            )
            return None

        inlier_ratio = len(inliers) / len(pts_3d)
        if inlier_ratio < self.min_inlier_ratio:
            logger.warning(
                f"{corr.src} -> {corr.dst}: low inlier ratio {inlier_ratio:.2f}"
            )
            return None

        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t

        return RelativePose(
            src=corr.src,
            dst=corr.dst,
            T_rel=T_rel,
            inliers=inliers.flatten(),
            num_inliers=int(len(inliers)),
            score=float(inlier_ratio),
            num_matches=int(len(corr.match_idx)),
        )
