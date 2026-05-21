"""Sequential global pose initializer.

Chains poses along the input ``frame_names`` order using the edge
``(frame_names[i], frame_names[i+1])`` if present. If an edge is missing,
the next frame inherits the previous world pose (matching the prior pipeline's
"PnP failed -> keep T_world" fallback), and is **not** reported as
unregistered — it still has a (degenerate) initial estimate that the optional
BA stage can refine.
"""

import logging
from typing import List, Tuple

import numpy as np

from ..types import GlobalPoses, RelativePoseMap

logger = logging.getLogger(__name__)


class SequentialInitializer:
    """Chain poses along ``frame_names`` order using forward edges only."""

    def initialize(
        self,
        frame_names: List[str],
        relative_poses: RelativePoseMap,
    ) -> Tuple[GlobalPoses, List[str]]:
        if not frame_names:
            return {}, []

        poses: GlobalPoses = {frame_names[0]: np.eye(4)}
        T_world = np.eye(4)
        missing_edges = 0

        for i in range(len(frame_names) - 1):
            src, dst = frame_names[i], frame_names[i + 1]
            rel = relative_poses.get((src, dst))
            if rel is None:
                logger.warning(f"Sequential init: no edge {src} -> {dst}")
                missing_edges += 1
                poses[dst] = T_world.copy()
                continue
            # T_world_dst = T_world_src @ inv(T_rel)
            T_world = T_world @ np.linalg.inv(rel.T_rel)
            poses[dst] = T_world.copy()

        if missing_edges:
            logger.warning(
                f"Sequential init: {missing_edges} missing edges; "
                "downstream frames may be drifted"
            )

        logger.info(f"Sequential init: {len(poses)}/{len(frame_names)} placed")
        return poses, []
