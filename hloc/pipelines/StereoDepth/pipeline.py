"""
Stereo Depth Pipeline: hloc feature matching + depth-based PnP + GTSAM BA

This pipeline uses:
1. hloc for feature extraction (SuperPoint) and matching (LightGlue)
2. Pre-computed depth maps for 3D point recovery
3. PnP + RANSAC for pose estimation
4. GTSAM Bundle Adjustment for global pose optimization
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict
import json

import cv2
import numpy as np
import h5py
from tqdm import tqdm

from hloc import extract_features, match_features
from hloc.utils.io import get_keypoints, get_matches

logger = logging.getLogger(__name__)

# Check if GTSAM is available
try:
    import gtsam
    from gtsam import symbol_shorthand
    L = symbol_shorthand.L  # Landmark
    X = symbol_shorthand.X  # Pose
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    logger.warning("GTSAM not available, BA will be skipped")


# Feature extraction config - SuperPoint
EXTRACT_CONF = {
    "output": "feats-superpoint-n4096-r1600",
    "model": {
        "name": "superpoint",
        "nms_radius": 3,
        "max_keypoints": 4096,
    },
    "preprocessing": {
        "grayscale": True,
        "resize_max": 1600,  # Resize for speed, original is 4032x3036
    },
}

# Matching config - LightGlue
MATCH_CONF = {
    "output": "matches-lightglue",
    "model": {
        "name": "lightglue",
        "features": "superpoint",
    },
}


def load_camera_intrinsics(k_txt_path: Path) -> tuple:
    """Load camera intrinsics from K.txt file.

    Format:
        Line 1: fx 0 cx 0 fy cy 0 0 1 (9 values, row-major 3x3)
        Line 2: baseline in meters

    Returns:
        K: 3x3 intrinsic matrix
        baseline: stereo baseline in meters
    """
    with open(k_txt_path, 'r') as f:
        lines = f.readlines()

    k_values = list(map(float, lines[0].strip().split()))
    K = np.array(k_values).reshape(3, 3)
    baseline = float(lines[1].strip())

    return K, baseline


def load_depth_map(depth_path: Path) -> np.ndarray:
    """Load depth map from .npy file (in meters)."""
    return np.load(depth_path)


def get_3d_from_depth(keypoints: np.ndarray, depth_map: np.ndarray,
                      K: np.ndarray) -> tuple:
    """Back-project 2D keypoints to 3D using depth map.

    Args:
        keypoints: Nx2 array of (u, v) pixel coordinates in original image size
        depth_map: HxW depth map in meters
        K: 3x3 camera intrinsic matrix

    Returns:
        points_3d: Nx3 array of 3D points in camera frame
        valid_mask: N boolean array indicating valid depth
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    N = len(keypoints)
    points_3d = np.zeros((N, 3), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    H, W = depth_map.shape

    for i, (u, v) in enumerate(keypoints):
        # Round to nearest pixel
        u_int, v_int = int(round(u)), int(round(v))

        # Bounds check
        if 0 <= u_int < W and 0 <= v_int < H:
            z = depth_map[v_int, u_int]

            # Valid depth check (positive and reasonable range)
            if 0.1 < z < 50.0:  # 0.1m to 50m
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points_3d[i] = [x, y, z]
                valid_mask[i] = True

    return points_3d, valid_mask


def estimate_pose_pnp(pts_3d: np.ndarray, pts_2d: np.ndarray,
                      K: np.ndarray, dist_coeffs: np.ndarray = None) -> tuple:
    """Estimate camera pose using PnP + RANSAC.

    Args:
        pts_3d: Nx3 3D points in previous frame's camera coordinate
        pts_2d: Nx2 2D points in current frame
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (optional)

    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        inliers: Inlier indices
        success: Whether pose estimation succeeded
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)

    if len(pts_3d) < 6:
        return None, None, None, False

    # Ensure correct data types for OpenCV
    pts_3d = np.ascontiguousarray(pts_3d, dtype=np.float64)
    pts_2d = np.ascontiguousarray(pts_2d, dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.float64)
    dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float64)

    # Use P3P with RANSAC
    # For high-resolution images (4032x3036), use larger reprojection threshold
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, dist_coeffs,
        iterationsCount=2000,
        reprojectionError=8.0,  # Increased for high-res images
        confidence=0.99,
        flags=cv2.SOLVEPNP_P3P
    )

    if not success or inliers is None or len(inliers) < 6:
        return None, None, None, False

    # Convert to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()

    # Validate pose (check for unreasonable motion)
    translation_magnitude = np.linalg.norm(t)
    if translation_magnitude > 10.0:  # Max 10m per frame (relaxed)
        logger.warning(f"Large translation detected: {translation_magnitude:.2f}m")
        return None, None, None, False

    inlier_ratio = len(inliers) / len(pts_3d)
    if inlier_ratio < 0.15:  # Lowered threshold
        logger.warning(f"Low inlier ratio: {inlier_ratio:.2f}")
        return None, None, None, False

    return R, t, inliers.flatten(), True


def pose_to_gtsam(T: np.ndarray) -> "gtsam.Pose3":
    """Convert 4x4 transformation matrix to GTSAM Pose3."""
    R = T[:3, :3]
    t = T[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))


def gtsam_to_pose(pose3: "gtsam.Pose3") -> np.ndarray:
    """Convert GTSAM Pose3 to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = pose3.rotation().matrix()
    T[:3, 3] = pose3.translation()
    return T


def run_bundle_adjustment(
    poses: dict,
    observations: list,
    K: np.ndarray,
    fix_first_pose: bool = True
) -> dict:
    """Run Bundle Adjustment using GTSAM.

    Args:
        poses: Dictionary mapping image name to 4x4 pose matrix
        observations: List of observation dicts with:
            - 'frame0': source frame name
            - 'frame1': target frame name
            - 'pts_3d': Nx3 3D points in frame0's camera coordinate
            - 'pts_2d': Nx2 2D observations in frame1
            - 'inliers': indices of inlier matches
        K: 3x3 camera intrinsic matrix
        fix_first_pose: Whether to fix the first pose

    Returns:
        optimized_poses: Dictionary with optimized poses
    """
    if not GTSAM_AVAILABLE:
        logger.warning("GTSAM not available, skipping BA")
        return poses

    logger.info("Running Bundle Adjustment with GTSAM...")

    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Create initial estimates
    initial = gtsam.Values()

    # Map frame names to indices
    frame_names = list(poses.keys())
    frame_to_idx = {name: i for i, name in enumerate(frame_names)}

    # Add initial pose estimates
    for name, T in poses.items():
        idx = frame_to_idx[name]
        pose3 = pose_to_gtsam(T)
        initial.insert(X(idx), pose3)

    # Prior noise model for first pose (very tight if fixed)
    if fix_first_pose:
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])  # rad, rad, rad, m, m, m
        )
        graph.add(gtsam.PriorFactorPose3(X(0), pose_to_gtsam(poses[frame_names[0]]), prior_noise))

    # Camera calibration for GTSAM
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cal = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    # Noise model for projection (pixel noise)
    projection_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)  # 2 pixel std

    # Add projection factors
    num_factors = 0
    for obs in observations:
        if 'pts_3d' not in obs or 'pts_2d' not in obs:
            continue

        frame0, frame1 = obs['frame0'], obs['frame1']
        if frame0 not in frame_to_idx or frame1 not in frame_to_idx:
            continue

        idx0 = frame_to_idx[frame0]
        idx1 = frame_to_idx[frame1]

        pts_3d = obs['pts_3d']
        pts_2d = obs['pts_2d']
        inliers = obs.get('inliers', np.arange(len(pts_3d)))

        # Get pose of frame0 to transform 3D points to world frame
        T0 = poses[frame0]

        # Add factors for inlier observations
        for i in inliers[:100]:  # Limit to 100 points per pair for speed
            if i >= len(pts_3d):
                continue

            # 3D point in frame0's camera coordinate
            pt_cam0 = pts_3d[i]

            # Transform to world frame
            pt_world = (T0 @ np.array([*pt_cam0, 1.0]))[:3]

            # 2D observation in frame1
            pt_2d = pts_2d[i]

            # Create projection factor
            # This factor measures: "given camera pose X(idx1), the 3D point pt_world
            # should project to pt_2d"
            try:
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(pt_2d[0], pt_2d[1]),
                    projection_noise,
                    X(idx1),
                    L(num_factors),  # Use unique landmark ID
                    cal
                )
                graph.add(factor)

                # Add landmark with strong prior (since we know it from depth)
                landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.05)  # 5cm std
                graph.add(gtsam.PriorFactorPoint3(
                    L(num_factors),
                    gtsam.Point3(pt_world[0], pt_world[1], pt_world[2]),
                    landmark_noise
                ))
                initial.insert(L(num_factors), gtsam.Point3(pt_world[0], pt_world[1], pt_world[2]))

                num_factors += 1
            except Exception as e:
                continue

    logger.info(f"Added {num_factors} projection factors")

    if num_factors < 10:
        logger.warning("Too few factors for BA, skipping")
        return poses

    # Optimize
    try:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        # Extract optimized poses
        optimized_poses = {}
        for name in frame_names:
            idx = frame_to_idx[name]
            pose3 = result.atPose3(X(idx))
            optimized_poses[name] = gtsam_to_pose(pose3)

        # Report improvement
        initial_error = graph.error(initial)
        final_error = graph.error(result)
        logger.info(f"BA complete: error {initial_error:.2f} -> {final_error:.2f}")

        return optimized_poses

    except Exception as e:
        logger.error(f"BA failed: {e}")
        return poses


def run_pipeline(data_dir: Path, output_dir: Path,
                 visualize: bool = False,
                 concatenate_pcd: bool = True,
                 voxel_size: float = 0.01,
                 run_ba: bool = False) -> dict:
    """Run the stereo depth pipeline.

    Args:
        data_dir: Directory containing timestamp folders
        output_dir: Output directory for results
        visualize: Whether to show visualizations
        concatenate_pcd: Whether to concatenate point clouds
        voxel_size: Voxel size for point cloud downsampling
        run_ba: Whether to run Bundle Adjustment

    Returns:
        results: Dictionary with poses and statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all timestamp folders
    timestamp_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])

    if len(timestamp_dirs) < 2:
        raise ValueError(f"Need at least 2 frames, found {len(timestamp_dirs)}")

    logger.info(f"Found {len(timestamp_dirs)} frames")

    # Step 1: Prepare images for hloc
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    frame_info = {}  # Map image name -> original data
    image_list = []

    for ts_dir in timestamp_dirs:
        left_img = ts_dir / "rect_left.jpg"
        if not left_img.exists():
            logger.warning(f"Missing rect_left.jpg in {ts_dir}")
            continue

        # Create symlink or copy to images dir
        img_name = f"{ts_dir.name}.jpg"
        dst_path = images_dir / img_name

        if not dst_path.exists():
            import shutil
            shutil.copy(left_img, dst_path)

        # Load camera intrinsics
        k_txt = ts_dir / "K.txt"
        K, baseline = load_camera_intrinsics(k_txt)

        frame_info[img_name] = {
            "timestamp_dir": ts_dir,
            "K": K,
            "baseline": baseline,
            "depth_path": ts_dir / "depth_meter.npy",
            "cloud_path": ts_dir / "cloud.ply",
        }
        image_list.append(img_name)

    logger.info(f"Prepared {len(image_list)} images for feature extraction")

    # Step 2: Extract features using hloc
    features_path = output_dir / "features.h5"

    logger.info("Extracting features with SuperPoint...")
    extract_features.main(
        conf=EXTRACT_CONF,
        image_dir=images_dir,
        export_dir=output_dir,
        feature_path=features_path,
    )

    # Step 3: Create pairs (sequential pairs for VO)
    pairs_path = output_dir / "pairs.txt"
    with open(pairs_path, 'w') as f:
        for i in range(len(image_list) - 1):
            f.write(f"{image_list[i]} {image_list[i+1]}\n")

    logger.info(f"Created {len(image_list) - 1} sequential pairs")

    # Step 4: Match features using hloc
    matches_path = output_dir / "matches.h5"

    logger.info("Matching features with LightGlue...")
    match_features.main(
        conf=MATCH_CONF,
        pairs=pairs_path,
        features=features_path,
        export_dir=output_dir,
        matches=matches_path,
    )

    # Step 5: Pose estimation with depth-based PnP
    logger.info("Estimating poses with depth-based PnP...")

    # Note: hloc automatically scales keypoint coordinates back to original image size
    # (see extract_features.py line 274), so no manual scaling is needed

    # Initialize poses (first frame is identity)
    poses = {image_list[0]: np.eye(4)}
    T_world = np.eye(4)  # Cumulative pose

    pose_results = []
    ba_observations = []  # Collect observations for BA

    # Store K matrix for BA
    sample_K = frame_info[image_list[0]]["K"]

    for i in tqdm(range(len(image_list) - 1), desc="PnP"):
        img0, img1 = image_list[i], image_list[i + 1]

        # Get keypoints (get_keypoints takes path and image name)
        kpts0 = get_keypoints(features_path, img0)
        kpts1 = get_keypoints(features_path, img1)

        # Get matches (returns (matches, scores) where matches is Nx2 array)
        try:
            matches, scores = get_matches(matches_path, img0, img1)
        except ValueError as e:
            logger.warning(f"No matches found between {img0} and {img1}: {e}")
            poses[img1] = T_world.copy()
            pose_results.append({
                "frame0": img0, "frame1": img1,
                "success": False, "reason": "no_matches"
            })
            continue

        num_matches = len(matches)
        if num_matches < 10:
            logger.warning(f"Too few matches between {img0} and {img1}: {num_matches}")
            poses[img1] = T_world.copy()
            pose_results.append({
                "frame0": img0, "frame1": img1,
                "success": False, "reason": "too_few_matches"
            })
            continue

        # matches is Nx2 array: matches[i] = [idx0, idx1]
        matched_kpts0 = kpts0[matches[:, 0]]
        matched_kpts1 = kpts1[matches[:, 1]]

        # Load depth for frame 0
        info0 = frame_info[img0]
        depth0 = load_depth_map(info0["depth_path"])
        K = info0["K"]

        # Get 3D points from depth (keypoints are already in original image coordinates)
        pts_3d, depth_valid = get_3d_from_depth(matched_kpts0, depth0, K)

        if depth_valid.sum() < 10:
            logger.warning(f"Too few valid depth points: {depth_valid.sum()}")
            poses[img1] = T_world.copy()
            pose_results.append({
                "frame0": img0, "frame1": img1,
                "success": False, "reason": "too_few_depth"
            })
            continue

        # Filter to only points with valid depth
        pts_3d_valid = pts_3d[depth_valid]
        pts_2d_valid = matched_kpts1[depth_valid]  # Already in original coords

        # PnP pose estimation
        R, t, inliers, success = estimate_pose_pnp(pts_3d_valid, pts_2d_valid, K)

        if not success:
            logger.warning(f"PnP failed for {img0} -> {img1}")
            poses[img1] = T_world.copy()
            pose_results.append({
                "frame0": img0, "frame1": img1,
                "success": False, "reason": "pnp_failed"
            })
            continue

        # Relative pose (from frame0 to frame1)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t

        # Accumulate pose: T_world_new = T_world @ inv(T_rel)
        # Because T_rel transforms points from frame0 to frame1
        T_world = T_world @ np.linalg.inv(T_rel)
        poses[img1] = T_world.copy()

        pose_results.append({
            "frame0": img0, "frame1": img1,
            "success": True,
            "num_matches": num_matches,
            "num_depth_valid": int(depth_valid.sum()),
            "num_inliers": int(len(inliers)),
            "translation": np.linalg.norm(t),
        })

        # Collect observation for BA
        ba_observations.append({
            "frame0": img0,
            "frame1": img1,
            "pts_3d": pts_3d_valid,
            "pts_2d": pts_2d_valid,
            "inliers": inliers,
        })

        logger.debug(f"{img0} -> {img1}: matches={num_matches}, "
                    f"depth_valid={depth_valid.sum()}, inliers={len(inliers)}, "
                    f"t={np.linalg.norm(t):.3f}m")

    # Step 5.5: Bundle Adjustment (optional)
    if run_ba and GTSAM_AVAILABLE and len(ba_observations) > 0:
        poses = run_bundle_adjustment(poses, ba_observations, sample_K)

    # Save poses
    poses_file = output_dir / "poses.txt"
    with open(poses_file, 'w') as f:
        for img_name, T in poses.items():
            f.write(f"# {img_name}\n")
            for row in T:
                f.write(" ".join(f"{v:.8f}" for v in row) + "\n")

    logger.info(f"Saved poses to {poses_file}")

    # Save pose results
    results_file = output_dir / "pose_results.json"
    with open(results_file, 'w') as f:
        json.dump(pose_results, f, indent=2)

    # Step 6: Concatenate point clouds (optional)
    if concatenate_pcd:
        logger.info("Concatenating point clouds...")
        concatenate_point_clouds(frame_info, poses, output_dir / "combined.ply", voxel_size)

    return {
        "num_frames": len(image_list),
        "num_successful": sum(1 for r in pose_results if r.get("success", False)),
        "poses": poses,
        "pose_results": pose_results,
    }


def concatenate_point_clouds(frame_info: dict, poses: dict, output_path: Path,
                             voxel_size: float = 0.01):
    """Concatenate point clouds using estimated poses.

    Args:
        frame_info: Dictionary mapping image name to frame info
        poses: Dictionary mapping image name to 4x4 pose matrix
        output_path: Output PLY file path
        voxel_size: Voxel size for downsampling (0 to disable)
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        logger.error("plyfile not installed, skipping point cloud concatenation")
        return

    all_points = []
    all_colors = []

    for img_name, info in tqdm(frame_info.items(), desc="Concatenating PCDs"):
        cloud_path = info["cloud_path"]
        if not cloud_path.exists():
            logger.warning(f"Missing point cloud: {cloud_path}")
            continue

        if img_name not in poses:
            logger.warning(f"No pose for {img_name}")
            continue

        # Load point cloud
        plydata = PlyData.read(str(cloud_path))
        vertex = plydata['vertex']

        # Extract XYZ
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        points = np.column_stack([x, y, z])

        # Extract colors if available
        has_color = 'red' in vertex.data.dtype.names
        if has_color:
            r = vertex['red']
            g = vertex['green']
            b = vertex['blue']
            colors = np.column_stack([r, g, b])
        else:
            colors = np.zeros((len(points), 3), dtype=np.uint8)

        # Transform to world frame: P_world = T @ [P_camera; 1]
        T = poses[img_name]
        points_h = np.column_stack([points, np.ones(len(points))])
        points_world = (T @ points_h.T).T[:, :3]

        all_points.append(points_world)
        all_colors.append(colors)

    if not all_points:
        logger.error("No point clouds to concatenate")
        return

    # Combine all points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)

    logger.info(f"Total points before downsampling: {len(combined_points)}")

    # Simple voxel downsampling if requested
    if voxel_size > 0:
        logger.info(f"Downsampling with voxel size {voxel_size}m...")
        # Quantize points to voxel grid
        voxel_indices = (combined_points / voxel_size).astype(np.int32)
        # Use unique voxels (keep first point in each voxel)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        combined_points = combined_points[unique_idx]
        combined_colors = combined_colors[unique_idx]
        logger.info(f"Points after downsampling: {len(combined_points)}")

    # Save as PLY
    vertex_data = np.zeros(len(combined_points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = combined_points[:, 0]
    vertex_data['y'] = combined_points[:, 1]
    vertex_data['z'] = combined_points[:, 2]
    vertex_data['red'] = combined_colors[:, 0]
    vertex_data['green'] = combined_colors[:, 1]
    vertex_data['blue'] = combined_colors[:, 2]

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=True).write(str(output_path))

    logger.info(f"Saved combined point cloud to {output_path} "
                f"({len(combined_points)} points)")


def main():
    parser = argparse.ArgumentParser(
        description="Stereo Depth Pipeline: hloc matching + depth-based PnP"
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Directory containing timestamp folders with stereo data"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Output directory (default: data_dir/hloc_output)"
    )
    parser.add_argument(
        "--no_pcd", action="store_true",
        help="Skip point cloud concatenation"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.01,
        help="Voxel size for point cloud downsampling (default: 0.01m)"
    )
    parser.add_argument(
        "--ba", action="store_true",
        help="Run Bundle Adjustment using GTSAM"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.output_dir is None:
        args.output_dir = args.data_dir / "hloc_output"

    results = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        concatenate_pcd=not args.no_pcd,
        voxel_size=args.voxel_size,
        run_ba=args.ba,
    )

    logger.info(f"Pipeline complete: {results['num_successful']}/{results['num_frames']-1} "
                f"successful pose estimates")


if __name__ == "__main__":
    main()
