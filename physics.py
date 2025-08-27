import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

@dataclass
class Point3D:
    id: int
    x: float
    y: float
    z: float


@dataclass
class Trajectory:
    id: int
    points: list[Point3D]


def load_points(path: str, key: str = "3d_point") -> list[Point3D]:
    """
    Load 3D points from a JSON annotations file.

    Returns list of 3D points skipping null / invalid entries.
    """

    with open(path, "r") as f:
        data = json.load(f)

    pts: list[Point3D] = []
    annotations = data.get("annotations", [])
    for ann in annotations:
        p = ann.get(key, None)
        if p is not None:
            pts.append(Point3D(id=len(pts), x=float(p[0]), y=float(p[1]), z=float(p[2])))
    return pts

def compute_angle_between(vel1: np.ndarray, vel2: np.ndarray) -> float:
    """
    Return angle between two velocity vectors.
    """
    n1 = np.linalg.norm(vel1)
    n2 = np.linalg.norm(vel2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0

    cos_angle = np.dot(vel1, vel2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def compute_derivatives(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def compute_curvatures(velocities: np.ndarray, accelerations: np.ndarray) -> np.ndarray:
    """
    Compute curvature given velocity and acceleration vectors.
    """
    cross = np.cross(velocities, accelerations)
    num = np.linalg.norm(cross, axis=1)
    denom = (np.linalg.norm(velocities, axis=1) ** 3) + 1e-12
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
    min_length: int = 3,                           # Minimum trajectory length
    sigma: float = 0.5,                            # Gaussian smoothing sigma
    thr_angle: float = np.pi / 6,                  # Directional change threshold
    k_dist: float = 3.0,                           # Distance threshold
    k_acc: float = 4.0,                            # Acceleration threshold
    k_curv: float = 2.5,                           # Curvature threshold
    speed_tolerance: float = 0.4,                  # Speed tolerance for minimum speed detection
    spike_ratio: float = 1.5,                      # Spike ratio for position discontinuities
    min_evidences: int = 3,                        # Minimum evidences to consider a break
    filter_meaningful_trajectories: bool = True,   # Filter only meaningful trajectories
    verbose: bool = False,
) -> List[Trajectory]:
    """
    Detect trajectories from a list of 3D points using physics-based and geometry-based heuristics.
    Returns list of Trajectory objects.
    """

    if not points:
        raise ValueError("No points provided.")

    n = len(points)
    if n < min_length:
        raise ValueError("Insufficient points to detect trajectories.")

    # Store the points also in Numpy format for later calculations
    np_points = np.array([(p.x, p.y, p.z) for p in points])

    # Prepare arrays
    xs = np_points[:, 0]
    ys = np_points[:, 1]
    zs = np_points[:, 2]

    # Geometric variables
    distances = np.linalg.norm(np.diff(np_points, axis=0), axis=1) # Pair-wise Euclidean distances
    xs = gaussian_filter1d(xs, sigma=sigma, mode="nearest") # Apply 
    ys = gaussian_filter1d(ys, sigma=sigma, mode="nearest")
    zs = gaussian_filter1d(zs, sigma=sigma, mode="nearest")

    # Motion variables
    velocities, speeds, accelerations, acc_magnitudes = compute_derivatives(xs, ys, zs)
    med_speed = np.median(speeds)
    curvatures = compute_curvatures(velocities, accelerations)

    # Thresholds
    thr_dist = compute_threshold(distances, k_dist)
    thr_acc = compute_threshold(acc_magnitudes, k_acc)
    thr_curv = compute_threshold(curvatures, k_curv)

    # Tolerances
    eps = 1e-12

    # candidate break points alongside with reasons
    candidate_breaks: List[Tuple[int, List[str]]] = []

    # Start and end are excluded as they are for sure break points
    for i in range(1, n - 2):
        # reasons array to store all the motivations for a point to be picked
        reasons = []

        # Evidence 1: local minima
        is_local_min = (zs[i] < zs[i-1] - eps) and (zs[i] < zs[i+1] - eps)

        # Evidence 2: local speed minima
        is_speed_min = (speeds[i] < speeds[i-1] - eps) and (speeds[i] < speeds[i+1] - eps) and (speeds[i] < speed_tolerance * med_speed)

        # Evidence 3: local acceleration maxima
        is_high_accel = (acc_magnitudes[i] > acc_magnitudes[i-1]) and (acc_magnitudes[i] > acc_magnitudes[i+1]) and (acc_magnitudes[i] > thr_acc)

        # Evidence 4: direction change (angle between velocity vectors)
        theta = compute_angle_between(velocities[i-1], velocities[i])
        angle_threshold = thr_angle
        if zs[i] > np.percentile(zs, 75):  # In upper 25% of heights
            angle_threshold = thr_angle * 1.5  # Require sharper turns
        is_angle_change = theta > angle_threshold

        # Evidence 5: local position discontinuites (sudden jumps respect the adjacent values)
        d = distances[i-1]
        prev_distance = distances[i-2] if (i-2) >= 0 else d
        next_distance = distances[i] if (i) < len(distances) else d
        local_spike = (d > prev_distance * spike_ratio) and (d > next_distance * spike_ratio)
        global_outlier = d > thr_dist
        is_pos_discont = local_spike or global_outlier

        # Evidence 6: high curvature
        is_high_curvature = curvatures[i] > thr_curv

        # Collect evidence
        evidence_count = 0
        if is_local_min:
            evidence_count += min_evidences - 1 # Quite strong evidence, so higher weight
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

        # Decision logic
        if evidence_count >= min_evidences:
            last_candidate = candidate_breaks[-1][0] if candidate_breaks else None
            if last_candidate:
                if (i - last_candidate) >= min_length:
                    candidate_breaks.append((i, reasons))
            else:
                candidate_breaks.append((i, reasons))

    if not candidate_breaks:
        break_indices = {0, n - 1} # Single trajectory scenario
    else:
        break_indices = {0} | {c[0] for c in candidate_breaks} | {n - 1} # Multiple trajectories scenario
        break_indices = sorted(break_indices)

    # Build trajectories from segments
    trajectories: List[Trajectory] = []
    for start, end in zip(break_indices, break_indices[1:]):
        trajectories.append(Trajectory(id=len(trajectories), points=points[start:end+1]))

    # Filter out trajectories meaningless trajectories
    # (Usually the first dribbles before the ball serve)
    if filter_meaningful_trajectories:
        for idx, traj in enumerate(trajectories):
            is_inside_field = False
            if traj.points[0].x >= 0 and traj.points[0].x <= 18:
                if traj.points[0].y >= 0 and traj.points[0].y <= 9:
                    is_inside_field = True
            if not is_inside_field:
                if traj.points[-1].x >= 0 and traj.points[-1].x <= 18:
                    if traj.points[-1].y >= 0 and traj.points[-1].y <= 9:
                        is_inside_field = True
            if is_inside_field:
                trajectories = trajectories[idx:]
                break
        for idx, traj in enumerate(trajectories):
            traj.id = idx

    if verbose:
        for traj in trajectories:
            start = traj.points[0].id
            end = traj.points[-1].id
            print(f'Trajectory {traj.id}:')
            start_reasons = next((reasons for (idx, reasons) in candidate_breaks if idx == start), ["start point"])
            end_reasons = next((reasons for (idx, reasons) in candidate_breaks if idx == end), ["end point"])
            print(f"  - {start}: {start_reasons}")
            print(f"  - {end}: {end_reasons}")

    return trajectories

def plot_trajectories(trajectories: List[Trajectory], plot_points: bool = False, plot_markers: bool = False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points faintly for context
    if plot_points:
        all_pts = np.array([[p.x, p.y, p.z] for traj in trajectories for p in traj.points])
        if all_pts.size > 0:
            ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], s=5, alpha=0.15, color="gray", label="All points")

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
        ax.text(traj_pts[0, 0], traj_pts[0, 1], traj_pts[0, 2] + 0.2, str(traj_id), color=color, fontsize=9, weight="bold")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    import glob

    for i in range(10):
        file = glob.glob(f"C:/Users/Matteo/Downloads/3D-prime10/3D/{i+1}/3d_points.json")
        pts = load_points(file[0]) if file else []
        trajectories = detect_trajectories(pts, verbose=True)
        plot_trajectories(trajectories, plot_points=True, plot_markers=True)
