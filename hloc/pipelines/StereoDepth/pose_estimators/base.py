"""Protocol for per-pair pose estimation."""

from typing import Optional, Protocol, runtime_checkable

from ..types import PairCorrespondence, RelativePose


@runtime_checkable
class PairPoseEstimator(Protocol):
    """Estimate a directed relative pose from one pair of correspondences.

    Implementations must return a ``RelativePose`` whose ``T_rel`` follows the
    locked convention ``p_dst = T_rel @ p_src``. Return ``None`` if the
    estimate is unreliable (too few inliers, degenerate configuration, etc.).
    """

    def estimate(self, corr: PairCorrespondence) -> Optional[RelativePose]: ...
