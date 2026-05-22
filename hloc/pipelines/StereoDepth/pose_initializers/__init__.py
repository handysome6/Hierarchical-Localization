"""Global pose initializers.

Each initializer turns a (possibly sparse) directed map of pairwise relative
poses into a global pose set keyed by image name. Implementations differ in
how they choose seed frames, propagate poses, and handle missing edges.
"""

from .aruco import ArucoMarkerInitializer
from .base import GlobalPoseInitializer
from .from_json import JsonInitializer
from .incremental import IncrementalInitializer
from .sequential import SequentialInitializer

__all__ = [
    "GlobalPoseInitializer",
    "IncrementalInitializer",
    "SequentialInitializer",
    "JsonInitializer",
    "ArucoMarkerInitializer",
]
