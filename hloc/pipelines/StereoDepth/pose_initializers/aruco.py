"""ArUco-marker-based global pose initializer.

For each frame we detect ArUco markers in ``rect_left.jpg`` (with
``detectInvertedMarker=True`` so color-reversed markers are picked up too),
sample camera-frame XYZ at the four corner pixels from the depth map, and
build a per-frame index ``marker_id -> (4, 3) camera-frame XYZ``.

Pairwise pose between two frames that share one or more marker IDs is solved
by rigid 3D-3D Procrustes (Kabsch with translation) over all shared, valid
corner pairs. We then greedily chain frames into a single world frame, just
like :class:`IncrementalInitializer` does for PnP edges -- but weighted by
shared-corner count instead of PnP inliers.

This initializer ignores the ``RelativePoseMap`` argument it is handed; it
derives its own pairwise transforms from marker geometry alone.

Single-marker pairs (4 non-collinear coplanar points) are well-posed for
rigid alignment but we still log them so the user can tighten the threshold
if drift is observed.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..types import FrameInfo, GlobalPoses, RelativePoseMap

logger = logging.getLogger(__name__)


@dataclass
class _MarkerObs:
    """A single marker observation in one frame."""

    pixels: np.ndarray  # (4, 2) refined corner pixels
    xyz: np.ndarray  # (4, 3) camera-frame XYZ
    valid: np.ndarray  # (4,) bool, True where depth sample is valid


def _make_detector(dict_id: int) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.detectInvertedMarker = True
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def _sample_depth_patch(
    depth: np.ndarray, u: float, v: float, half: int = 1
) -> Optional[float]:
    """Sample a small patch around ``(u, v)`` and return the median of valid
    depth values. ``half=1`` -> 3x3 patch. Returns None if no valid samples."""
    H, W = depth.shape
    u0, v0 = int(round(u)), int(round(v))
    if u0 < 0 or v0 < 0 or u0 >= W or v0 >= H:
        return None
    u_lo, u_hi = max(0, u0 - half), min(W, u0 + half + 1)
    v_lo, v_hi = max(0, v0 - half), min(H, v0 + half + 1)
    patch = depth[v_lo:v_hi, u_lo:u_hi]
    mask = np.isfinite(patch) & (patch > 0)
    if not mask.any():
        return None
    return float(np.median(patch[mask]))


def _backproject(
    K: np.ndarray, u: float, v: float, z: float
) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def _kabsch(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Rigid 3D-3D alignment. Returns 4x4 T s.t. ``dst = T @ src``.

    Assumes ``src_pts`` and ``dst_pts`` are (N, 3) with N >= 3 non-collinear
    correspondences. Reflection is forbidden via the determinant sign trick.
    """
    src_c = src_pts.mean(axis=0)
    dst_c = dst_pts.mean(axis=0)
    P = src_pts - src_c
    Q = dst_pts - dst_c
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class ArucoMarkerInitializer:
    """Initialize global poses from ArUco-marker 3D-3D correspondences.

    Parameters
    ----------
    frame_info : dict
        ``{frame_name: FrameInfo}`` -- needed to read the rectified image,
        depth map, and intrinsics per frame.
    aruco_dict : int
        OpenCV dictionary id. Defaults to ``DICT_4X4_100`` because that's
        what publichouse2 uses; override for other datasets.
    min_shared_corners : int
        Minimum valid shared corners required to align a candidate frame to
        a reference. 4 = one marker's worth; 3 is the geometric minimum but
        marginal in the presence of depth noise.
    depth_patch_half : int
        Half-width of the median-filter patch used when sampling depth at a
        corner pixel. 1 -> 3x3 window. Set to 0 to use the exact pixel only.
    seed : str | None
        Frame to fix at the world origin. Defaults to the frame with the
        most valid corner observations at call time.
    """

    def __init__(
        self,
        frame_info: Dict[str, FrameInfo],
        aruco_dict: int = cv2.aruco.DICT_4X4_100,
        min_shared_corners: int = 4,
        depth_patch_half: int = 1,
        seed: Optional[str] = None,
    ):
        self.frame_info = frame_info
        self.detector = _make_detector(aruco_dict)
        self.min_shared_corners = min_shared_corners
        self.depth_patch_half = depth_patch_half
        self.seed = seed

    # ------------------------------------------------------------------
    # Per-frame detection.
    # ------------------------------------------------------------------

    def _observe_frame(self, name: str) -> Dict[int, _MarkerObs]:
        info = self.frame_info[name]
        img = cv2.imread(str(info.image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"ArUco init: cannot read image {info.image_path}")
            return {}
        if info.depth_path is None or not info.depth_path.exists():
            logger.warning(f"ArUco init: missing depth for {name}")
            return {}
        depth = np.load(info.depth_path)
        if depth.shape != img.shape[:2]:
            logger.warning(
                f"ArUco init: depth/image shape mismatch for {name} "
                f"({depth.shape} vs {img.shape[:2]}); skipping"
            )
            return {}

        corners, ids, _ = self.detector.detectMarkers(img)
        if ids is None or len(ids) == 0:
            return {}

        K = info.K
        obs: Dict[int, _MarkerObs] = {}
        for c, mid in zip(corners, ids.flatten().tolist()):
            # ``c`` is (1, 4, 2) -- four corners ordered TL, TR, BR, BL.
            pix = np.asarray(c, dtype=np.float64).reshape(4, 2)
            xyz = np.zeros((4, 3), dtype=np.float64)
            valid = np.zeros(4, dtype=bool)
            for i, (u, v) in enumerate(pix):
                z = _sample_depth_patch(depth, u, v, self.depth_patch_half)
                if z is None:
                    continue
                xyz[i] = _backproject(K, u, v, z)
                valid[i] = True
            obs[int(mid)] = _MarkerObs(pixels=pix, xyz=xyz, valid=valid)
        return obs

    def _detect_all(
        self, frame_names: List[str]
    ) -> Dict[str, Dict[int, _MarkerObs]]:
        all_obs: Dict[str, Dict[int, _MarkerObs]] = {}
        total_corners = 0
        for name in frame_names:
            if name not in self.frame_info:
                logger.warning(f"ArUco init: no FrameInfo for {name}; skipping")
                continue
            obs = self._observe_frame(name)
            all_obs[name] = obs
            total_corners += sum(int(o.valid.sum()) for o in obs.values())
        logger.info(
            f"ArUco init: detected markers in "
            f"{sum(1 for v in all_obs.values() if v)}/{len(frame_names)} frames "
            f"({total_corners} valid corners total)"
        )
        return all_obs

    # ------------------------------------------------------------------
    # Pairwise alignment.
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_corners(
        obs_a: Dict[int, _MarkerObs], obs_b: Dict[int, _MarkerObs]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Stack camera-frame XYZ for all shared markers' valid corners.

        Returns ``(pts_a, pts_b, n)`` where ``pts_a`` and ``pts_b`` are
        ``(n, 3)`` and aligned row-by-row.
        """
        a_list: List[np.ndarray] = []
        b_list: List[np.ndarray] = []
        for mid, oa in obs_a.items():
            ob = obs_b.get(mid)
            if ob is None:
                continue
            both = oa.valid & ob.valid
            if not both.any():
                continue
            a_list.append(oa.xyz[both])
            b_list.append(ob.xyz[both])
        if not a_list:
            return np.empty((0, 3)), np.empty((0, 3)), 0
        pts_a = np.vstack(a_list)
        pts_b = np.vstack(b_list)
        return pts_a, pts_b, pts_a.shape[0]

    def _solve_pair(
        self, obs_src: Dict[int, _MarkerObs], obs_dst: Dict[int, _MarkerObs]
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Return ``(T_rel, n_shared)`` where ``p_dst = T_rel @ p_src``."""
        pts_src, pts_dst, n = self._shared_corners(obs_src, obs_dst)
        if n < self.min_shared_corners:
            return None
        # Reject collinear configurations -- a single marker on a degenerate
        # diagonal would slip through n>=3 but break Kabsch.
        centered = pts_src - pts_src.mean(axis=0)
        s = np.linalg.svd(centered, compute_uv=False)
        if s[1] < 1e-4 * s[0]:
            logger.debug("ArUco pair: degenerate (collinear) source points")
            return None
        T = _kabsch(pts_src, pts_dst)
        return T, n

    # ------------------------------------------------------------------
    # Public protocol entry-point.
    # ------------------------------------------------------------------

    def initialize(
        self,
        frame_names: List[str],
        relative_poses: RelativePoseMap,
    ) -> Tuple[GlobalPoses, List[str]]:
        # relative_poses is intentionally unused; this initializer derives
        # pairwise transforms from marker geometry directly.
        del relative_poses

        if not frame_names:
            return {}, []

        all_obs = self._detect_all(frame_names)

        # Pick the seed: frame with the most valid corners (more anchor points
        # = a more stable origin for the world frame).
        def corner_count(name: str) -> int:
            return sum(int(o.valid.sum()) for o in all_obs.get(name, {}).values())

        seed_name = self.seed
        if seed_name is None:
            ranked = sorted(frame_names, key=corner_count, reverse=True)
            if not ranked or corner_count(ranked[0]) == 0:
                logger.warning("ArUco init: no frame has any valid corner; aborting")
                return {}, list(frame_names)
            seed_name = ranked[0]
        elif corner_count(seed_name) == 0:
            logger.warning(
                f"ArUco init: explicit seed {seed_name!r} has no valid corners"
            )

        poses: GlobalPoses = {seed_name: np.eye(4)}
        registered = {seed_name}
        unregistered = set(frame_names) - registered

        while unregistered:
            best_candidate: Optional[str] = None
            best_reference: Optional[str] = None
            best_T_rel: Optional[np.ndarray] = None
            best_n = 0

            for candidate in unregistered:
                obs_c = all_obs.get(candidate, {})
                if not obs_c:
                    continue
                for ref in registered:
                    obs_r = all_obs.get(ref, {})
                    if not obs_r:
                        continue
                    # Solve src=candidate, dst=reference so we can compose
                    # T_world_candidate = T_world_ref @ T_rel.
                    result = self._solve_pair(obs_c, obs_r)
                    if result is None:
                        continue
                    T_rel, n = result
                    if n > best_n:
                        best_n = n
                        best_candidate = candidate
                        best_reference = ref
                        best_T_rel = T_rel

            if best_candidate is None:
                break

            # p_ref = T_rel @ p_candidate => T_world_cand = T_world_ref @ T_rel.
            poses[best_candidate] = poses[best_reference] @ best_T_rel
            registered.add(best_candidate)
            unregistered.remove(best_candidate)
            logger.debug(
                f"ArUco init: registered {best_candidate} via {best_reference} "
                f"({best_n} shared corners)"
            )

        unreg_sorted = sorted(unregistered)
        logger.info(
            f"ArUco init: {len(registered)}/{len(frame_names)} placed "
            f"({len(unreg_sorted)} without marker overlap)"
        )
        return poses, unreg_sorted
