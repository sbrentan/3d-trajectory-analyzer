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

def derivative(arr: np.ndarray) -> np.ndarray:
    """
    Compute the first derivative d(arr)/dt (assuming uniform sampling).
    """
    return np.gradient(arr)

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Return angle in radians between 1D arrays v1 and v2.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10: 
        return 0.0
    cosv = np.dot(v1, v2) / (n1 * n2)
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.arccos(cosv)

def detect_trajectories(
    points: List[Point3D],
    min_length: int = 3,                    # minimum trajectory length (number of points)
    vz_thresh: float = 0.05,                # vertical velocity magnitude for sign change (m/s)
    angle_thresh: float = np.pi / 6,        # angle change threshold (radians)
    verbose: bool = False
) -> List[Trajectory]:
    """
    Detect trajectories using physics-informed, rule-based heuristics.

    Returns list of Trajectory objects.
    """
    if not points:
        return []

    n = len(points)
    if n < min_length:
        return [Trajectory(id=0, points=points)]

    # Prepare arrays
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)
    zs = np.array([p.z for p in points], dtype=float)

    # Prepare max/min values
    max_x = max(xs) - 1e-6
    max_y = max(ys) - 1e-6
    min_x = min(xs) + 1e-6
    min_y = min(ys) + 1e-6

    # Velocities
    vx = derivative(xs)  # Derivative of x values (i.e. horizontal velocity)
    vy = derivative(ys)  # Derivative of y values (i.e. horizontal velocity)
    vz = derivative(zs)  # Derivative of z values (i.e. vertical velocity)

    # Combine horizontal velocities into a 2D vector
    vxy = np.vstack((vx, vy)).T

    # candidate break indices
    candidate_breaks = []

    for i in range(1, n-1):
        reasons = []

        # Check 1: Local minima condition
        is_local_min = zs[i] < zs[i-1] and zs[i] < zs[i+1]

        # Check 2: Max/Min values condition
        is_max = xs[i] > max_x or ys[i] > max_y
        is_min = xs[i] < min_x or ys[i] < min_y

        # Check 3: ensure vertical velocity change (down -> up)
        window = max(1, min(2, i, n-2-i))  # ensure at least 1 to avoid empty slices
        vz_before = np.mean(vz[max(0, i-window):i])
        vz_after = np.mean(vz[i+1:min(n, i+1+window)])
        vz_sign_change = (vz_before < -vz_thresh) and (vz_after > vz_thresh)

        # Check 4: ensure angle variation between vertical velocity vectors
        v_before_z = np.array([0.0, 0.0, vz_before])
        v_after_z = np.array([0.0, 0.0, vz_after])
        vert_angle = angle_between(v_before_z, v_after_z)
        vert_angle_significant = vert_angle > angle_thresh

        # Check 5: ensure angle variation between horizontal velocity vectors
        horiz_angle = angle_between(vxy[i-1], vxy[i+1])
        horiz_angle_significant = horiz_angle > angle_thresh

        # Collect evidence
        evidence_count = 0
        if is_max or is_min:
            evidence_count += 1
            reasons.append("max_min_condition")
        if vz_sign_change:
            evidence_count += 1
            reasons.append("vz_sign_change")
        if vert_angle_significant:
            evidence_count += 1
            reasons.append("vert_angle_jump")
        if horiz_angle_significant:
            evidence_count += 1
            reasons.append("horiz_angle_jump")

        # Decision logic
        if is_local_min and evidence_count >= 1:
            reasons.insert(0, "local_minimum+weak_evidence")
            candidate_breaks.append((i, reasons))
        elif (is_max or is_min) and evidence_count >= 2:
            reasons.insert(0, "max_min_condition+medium_evidence")
            candidate_breaks.append((i, reasons))
        elif evidence_count >= 3:
            reasons.insert(0, "strong_evidence")
            candidate_breaks.append((i, reasons))

    if not candidate_breaks:
        break_indices = [0, n-1] # Single trajectory
    else:
        break_indices = [0] + [c[0] for c in candidate_breaks] + [n-1]
        break_indices = sorted(list(set(break_indices)))

    # Build trajectories from segments
    trajectories: List[Trajectory] = []
    for start, end in zip(break_indices, break_indices[1:]):
        if end - start + 1 >= min_length:
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
        start_idx = pts.index(traj.points[0])
        end_idx = pts.index(traj.points[-1])
        print(f"Trajectory {traj.id}: ({start_idx}, {end_idx})")
    plot_trajectories(trajectories, plot_points=True, plot_markers=True)
