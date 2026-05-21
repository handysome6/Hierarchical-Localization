"""Protocol for global pose initialization."""

from typing import List, Protocol, Tuple, runtime_checkable

from ..types import GlobalPoses, RelativePoseMap


@runtime_checkable
class GlobalPoseInitializer(Protocol):
    """Seed a ``GlobalPoses`` dict from a directed map of relative poses.

    Returns
    -------
    poses : GlobalPoses
        Frames that were successfully placed in a single common world frame.
        Each value is a 4x4 ``T_world_cam`` matrix.
    unregistered : list[str]
        Frame names from ``frame_names`` that could not be placed (no path
        through ``relative_poses`` to any registered frame, missing JSON entry,
        etc.). The orchestrator decides whether to drop them or fall back.
    """

    def initialize(
        self,
        frame_names: List[str],
        relative_poses: RelativePoseMap,
    ) -> Tuple[GlobalPoses, List[str]]: ...
