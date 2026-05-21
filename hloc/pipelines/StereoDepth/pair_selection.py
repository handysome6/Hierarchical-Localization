"""Pair-list generation policies.

Each policy maps a frame list to a list of directed image-name pairs that the
matching stage will compute. The list is **ordered**; downstream stages that
need both directions iterate explicitly.
"""

import logging
from typing import List, Literal, Tuple

from .types import FrameInfo

logger = logging.getLogger(__name__)


def generate_pairs(
    frames: List[FrameInfo],
    mode: Literal["exhaustive", "sequential"] = "exhaustive",
    max_exhaustive: int = 50,
) -> List[Tuple[str, str]]:
    """Build the pair list for matching.

    Parameters
    ----------
    frames : list[FrameInfo]
    mode : {"exhaustive", "sequential"}
        "exhaustive" emits all O(n^2) unordered pairs (better BA constraints
        and implicit loop closure). "sequential" emits only ``(i, i+1)``.
    max_exhaustive : int
        Hard ceiling; if ``mode="exhaustive"`` but ``len(frames) > max_exhaustive``,
        the policy falls back to sequential and logs a warning.
    """
    names = [f.name for f in frames]
    n = len(names)
    if n < 2:
        return []

    if mode == "exhaustive" and n > max_exhaustive:
        logger.warning(
            f"{n} frames exceeds max_exhaustive={max_exhaustive}; "
            "falling back to sequential pairing"
        )
        mode = "sequential"

    if mode == "exhaustive":
        pairs = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
        logger.info(f"Generated {len(pairs)} exhaustive pairs")
    elif mode == "sequential":
        pairs = [(names[i], names[i + 1]) for i in range(n - 1)]
        logger.info(f"Generated {len(pairs)} sequential pairs")
    else:
        raise ValueError(f"Unknown pair mode: {mode}")

    return pairs
