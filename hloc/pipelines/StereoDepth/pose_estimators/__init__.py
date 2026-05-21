"""Per-pair pose estimators.

Each estimator consumes a ``PairCorrespondence`` and returns a ``RelativePose``
(or ``None`` if estimation fails). New estimators (essential matrix, learned
regressor, etc.) should subclass / implement ``PairPoseEstimator`` and live
in their own module here.
"""

from .base import PairPoseEstimator
from .depth_pnp import DepthPnPEstimator

__all__ = ["PairPoseEstimator", "DepthPnPEstimator"]
