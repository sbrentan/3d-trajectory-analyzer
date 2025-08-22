import json
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Point3D:
    id: int
    x: float
    y: float
    z: float

@dataclass
class Trajectory:
    id: int
    points: List[Point3D]

def load_points(path: str, key: str = "3d_point") -> List[Point3D]:
    """
    Load 3D points from a JSON annotations file.
    Returns list of 3D points skipping null / invalid entries.
    """
    with open(path, "r") as f:
        data = json.load(f)

    pts: List[Point3D] = []
    anns = data.get("annotations", [])
    for idx, ann in enumerate(anns):
        p = ann.get(key, None)
        if p is not None:
            pts.append(Point3D(id=idx, x=float(p[0]), y=float(p[1]), z=float(p[2])))
    return pts

def compute_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Return angle between two velocity vectors.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    if n1 < 1e-6 or n2 < 1e-6: 
        return 0
    
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def compute_derivatives(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Compute velocity and acceleration given positions.
    """

    vx = np.gradient(xs, axis=0)
    vy = np.gradient(ys, axis=0)
    vz = np.gradient(zs, axis=0)

    velocities = np.column_stack([vx, vy, vz])
    speeds = np.linalg.norm(velocities, axis=1)

    ax = np.gradient(vx, axis=0)
    ay = np.gradient(vy, axis=0)
    az = np.gradient(vz, axis=0)

    accelerations = np.column_stack([ax, ay, az])
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)

    return velocities, speeds, accelerations, acc_magnitudes

def compute_curvatures(v: np.ndarray, a: np.ndarray) -> float:
    """
    Compute curvature given velocity and acceleration vectors.
    """
    cross = np.cross(v, a)
    num = np.linalg.norm(cross, axis=1)
    denom = (np.linalg.norm(v, axis=1) ** 3) + 1e-12
    return num / denom

def compute_threshold(arr: np.ndarray, k: float) -> float:
    """
    Compute a threshold value from the given array.
    """
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-12
    return med + k * mad

def detect_trajectories(
    points: List[Point3D],
    angle_thresh: float = np.pi / 4,        # angle change threshold (radians)
    k_dist: float = 4.0,                    # Distance threshold
    k_acc: float = 3.0,                     # Acceleration threshold
    k_curv: float = 1.0,                    # Curvature threshold
    spike_ratio: float = 3.0,               # Spike ratio for position discontinuities
    min_evidences: int = 3,                 # Minimum evidences to consider a break
    verbose: bool = False
) -> List[Trajectory]:
    """
    Detect trajectories using physics-informed, rule-based heuristics.

    Returns list of Trajectory objects.
    """

    if not points:
        raise ValueError("No points provided.")

    n = len(points)
    if n < 3: # 3 is minimum number of points
        raise ValueError("Insufficient points to detect trajectories.")
    
    np_points = np.array([(p.x, p.y, p.z) for p in points])

    # Prepare arrays
    xs = np_points[:, 0]
    ys = np_points[:, 1]
    zs = np_points[:, 2]

    # Motion variables
    velocities, speeds, accelerations, acc_magnitudes = compute_derivatives(xs, ys, zs)
    curvatures = compute_curvatures(velocities, accelerations)
    distances = np.linalg.norm(np.diff(np_points, axis=0), axis=1)
    med_speed = np.median(speeds)

    # Thresholds
    thr_dist = compute_threshold(distances, k_dist)
    thr_acc = compute_threshold(acc_magnitudes, k_acc)
    thr_curv = compute_threshold(curvatures, k_curv)

    # Tolerances
    min_speed_tolerance = 0.6
    speed_tolerance = 1e-3
    eps = 1e-12

    # candidate break indices
    candidate_breaks = []

    for i in range(1, n - 2):
        reasons = []

        # Evidence 1: local minima
        is_local_min = (zs[i] < zs[i-1] - eps) and (zs[i] < zs[i+1] - eps)

        # Evidence 2: local speed minima
        is_speed_min = (speeds[i] < speeds[i-1] - eps) and (speeds[i] < speeds[i+1] - eps) and (speeds[i] < min_speed_tolerance * med_speed)

        # Evidence 3: local acceleration maxima
        is_high_accel = acc_magnitudes[i] > acc_magnitudes[i-1] and acc_magnitudes[i] > acc_magnitudes[i+1] and acc_magnitudes[i] > thr_acc

        # Evidence 4: direction change (angle between velocity vectors)
        theta = compute_angle_between(velocities[i-1], velocities[i])
        is_angle_change = theta > angle_thresh

        # Evidence 5: local position discontinuites (sudden jumps respect the adjacent values)
        if (i - 1) < len(distances):
            d = distances[i-1]
            prev_distance = distances[i-2] if (i-2) >= 0 else d
            next_distance = distances[i] if (i) < len(distances) else d
            local_spike = (d > prev_distance * spike_ratio) and (d > next_distance * spike_ratio)
            global_outlier = d > thr_dist
        is_pos_discont = local_spike or global_outlier

        # Evidence 6: drop of speed near to zero
        is_speed_zero = speeds[i] < speed_tolerance * med_speed

        # Evidence 7: high curvature
        is_high_curvature = curvatures[i] > thr_curv    

        # Collect evidence
        evidence_count = 0
        if is_local_min:
            evidence_count += 2  # Quite strong evidence (but not always true)
            reasons.append("local_minimum")
        if is_speed_min:
            evidence_count += 1
            reasons.append("local_speed_minimum")
        if is_high_accel:
            evidence_count += 1
            reasons.append("high_acceleration")
        if is_angle_change:
            evidence_count += 1
            reasons.append("direction_change")
        if is_pos_discont:
            evidence_count += 1
            reasons.append("position_discontinuity")
        if is_high_curvature:
            evidence_count += 1
            reasons.append("high_curvature")
        if is_speed_zero:
            evidence_count += 1
            reasons.append("speed_zero")

        # Decision logic
        if evidence_count >= min_evidences:
            last_candidate = candidate_breaks[-1][0] if candidate_breaks else None
            if last_candidate and (i - last_candidate) < 5:
                new_index = (i + last_candidate) // 2
                candidate_breaks[-1] = (new_index, reasons)
            else:
                candidate_breaks.append((i, reasons))

    if not candidate_breaks:
        break_indices = {0, n - 1}
    else:
        break_indices = {0} | {c[0] for c in candidate_breaks} | {n - 1}
        break_indices = sorted(break_indices)

    # Build trajectories from segments
    trajectories: List[Trajectory] = []
    for start, end in zip(break_indices, break_indices[1:]):
        if end - start + 1 >= 3:
            trajectories.append(Trajectory(id=len(trajectories), points=points[start:end+1]))

    if verbose:
        # print candidate details
        print("Candidate break points (index: reasons):")
        for idx, reasons in candidate_breaks:
            print(f"  {idx}: {reasons}")

    return trajectories

def plot_trajectories(trajectories: List[Trajectory], plot_points: bool = False, plot_markers: bool = False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points faintly for context
    if plot_points:
        all_pts = np.array([[p.x, p.y, p.z] for traj in trajectories for p in traj.points])
        if all_pts.size > 0:
            ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                       s=5, alpha=0.15, color="gray", label="All points")

    # Predefined 10 bright colors
    base_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    for traj_id, traj in enumerate(trajectories):
        traj_pts = np.array([[p.x, p.y, p.z] for p in traj.points])
        if traj_pts.size == 0:
            continue
        color = base_colors[traj_id % len(base_colors)]
        ax.plot(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2],
                color=color, linewidth=2, alpha=0.9, label=f"Traj {traj_id}")

        # Start & End markers
        if plot_markers:
            ax.scatter(traj_pts[0, 0], traj_pts[0, 1], traj_pts[0, 2], c="green", s=60, marker="o")
            ax.scatter(traj_pts[-1, 0], traj_pts[-1, 1], traj_pts[-1, 2], c="red", s=60, marker="x")

        # Add trajectory ID
        ax.text(traj_pts[0, 0], traj_pts[0, 1], traj_pts[0, 2] + 0.2, str(traj_id),
                color=color, fontsize=9, weight="bold")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pts = load_points("3d_points.json")
    trajectories = detect_trajectories(pts, verbose=True)
    for traj in trajectories:
        start_idx = traj.points[0].id
        end_idx = traj.points[-1].id
        print(f"Trajectory {traj.id}: ({start_idx}, {end_idx})")
    plot_trajectories(trajectories, plot_points=True, plot_markers=True)
