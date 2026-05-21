"""Protocol for pose-graph optimization."""

from typing import List, Optional, Protocol, runtime_checkable

from ..types import GlobalPoses, PoseGraphEdge


@runtime_checkable
class PoseGraphOptimizer(Protocol):
    """Refine an initial pose set under a set of relative-pose constraints.

    Parameters
    ----------
    initial_poses : GlobalPoses
        Per-frame ``T_world_cam`` seeds. The optimizer is free to skip any
        frame not referenced by an edge — such frames will be passed through
        unchanged in the returned dict.
    edges : list[PoseGraphEdge]
        Directed pairwise constraints. ``edge.T_rel`` follows the locked
        convention ``p_dst = T_rel @ p_src``.
    anchor : str | None
        Frame to fix at its initial pose with a tight prior. If ``None`` the
        optimizer picks a reasonable default (typically the first key of
        ``initial_poses`` that is referenced by at least one edge).
    """

    def optimize(
        self,
        initial_poses: GlobalPoses,
        edges: List[PoseGraphEdge],
        anchor: Optional[str] = None,
    ) -> GlobalPoses: ...
