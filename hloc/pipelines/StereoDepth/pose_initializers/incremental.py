"""Greedy incremental global pose initializer (order-independent).

Algorithm:

1. Seed: place the first frame at the world origin.
2. Loop: at each step, pick the unregistered frame with the most inliers to
   any currently-registered frame. Place it using the edge with that frame.
3. Terminate when no unregistered frame has an edge into the registered set.

The traversal does not depend on the input order of ``frame_names`` beyond
the choice of seed, which matches the prior pipeline's behavior.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..types import GlobalPoses, RelativePoseMap

logger = logging.getLogger(__name__)


class IncrementalInitializer:
    """Seed all frames via greedy best-match propagation.

    Parameters
    ----------
    min_inliers : int
        Edges with fewer inliers are ignored during traversal. Defaults to 15
        to match the prior pipeline.
    seed : str | None
        Frame name to fix at the world origin. Defaults to ``frame_names[0]``
        at call time.
    """

    def __init__(self, min_inliers: int = 15, seed: Optional[str] = None):
        self.min_inliers = min_inliers
        self.seed = seed

    def initialize(
        self,
        frame_names: List[str],
        relative_poses: RelativePoseMap,
    ) -> Tuple[GlobalPoses, List[str]]:
        if not frame_names:
            return {}, []

        seed_name = self.seed if self.seed is not None else frame_names[0]
        if seed_name not in frame_names:
            raise ValueError(f"seed {seed_name!r} not in frame_names")

        poses: GlobalPoses = {seed_name: np.eye(4)}
        registered = {seed_name}
        unregistered = set(frame_names) - registered

        # Pre-filter edges by the inlier threshold to keep the inner loop tight.
        edges = {
            key: rel
            for key, rel in relative_poses.items()
            if rel.num_inliers >= self.min_inliers
        }

        while unregistered:
            best_candidate: Optional[str] = None
            best_reference: Optional[str] = None
            best_key: Optional[Tuple[str, str]] = None
            best_inliers = 0

            for candidate in unregistered:
                for ref in registered:
                    for key in ((ref, candidate), (candidate, ref)):
                        rel = edges.get(key)
                        if rel is None:
                            continue
                        if rel.num_inliers > best_inliers:
                            best_inliers = rel.num_inliers
                            best_candidate = candidate
                            best_reference = ref
                            best_key = key

            if best_candidate is None:
                preview = list(unregistered)[:5]
                ellipsis = "..." if len(unregistered) > 5 else ""
                logger.warning(
                    f"Cannot register {len(unregistered)} frame(s): "
                    f"{preview}{ellipsis}"
                )
                break

            rel = edges[best_key]
            T_ref = poses[best_reference]

            # T_rel always maps p_dst = T_rel @ p_src.
            # Hence T_world_dst = T_world_src @ inv(T_rel).
            if best_key[0] == best_reference:
                # edge: reference -> candidate
                T_candidate = T_ref @ np.linalg.inv(rel.T_rel)
            else:
                # edge: candidate -> reference, so reference is the dst.
                # T_world_ref = T_world_cand @ inv(T_rel)
                # => T_world_cand = T_world_ref @ T_rel
                T_candidate = T_ref @ rel.T_rel

            poses[best_candidate] = T_candidate
            registered.add(best_candidate)
            unregistered.remove(best_candidate)
            logger.debug(
                f"Registered {best_candidate} via {best_reference} "
                f"({best_inliers} inliers)"
            )

        logger.info(f"Incremental init: {len(registered)}/{len(frame_names)} placed")
        return poses, sorted(unregistered)
