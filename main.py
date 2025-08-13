import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

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
    anns = data.get("annotations", [])
    for idx, ann in enumerate(anns):
        p = ann.get(key, None)
        if p is not None:
            pts.append(Point3D(id=idx, x=float(p[0]), y=float(p[1]), z=float(p[2])))
    return pts

def detect_trajectories(points, min_length=3, threshold=0.001) -> list[Trajectory]:
    """
    Detect trajectories by:
    1. Splitting at obstacles in X and Y
    2. Splitting at local minima in Z
    """
    if not points:
        return []

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]

    # X and Y boundaries (walls)
    min_x = min(xs) + threshold
    max_x = max(xs) - threshold
    min_y = min(ys) + threshold
    max_y = max(ys) - threshold

    break_indices = [0]  # include first point
    for i in range(1, len(points) - 1):

        # Condition for Xs (for walls)
        if xs[i] < min_x or xs[i] > max_x:
            break_indices.append(i)

        # Condition for Ys (for walls)
        if ys[i] < min_y or ys[i] > max_y:
            break_indices.append(i)

        # Local minima condition for Zs (for parabolic motion)
        if zs[i] < zs[i-1] and zs[i] < zs[i+1]:
            break_indices.append(i)
    break_indices.append(len(points) - 1)  # include last point

    # Build trajectories
    trajectories: list[Trajectory] = []
    for start, end in zip(break_indices, break_indices[1:]):
        if end - start >= min_length:
            trajectories.append(Trajectory(id=len(trajectories), points=points[start:end+1]))

    return trajectories

def plot_trajectories(trajectories, plot_points=False, plot_markers=False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points faintly for context
    if plot_points:
        all_pts = np.array([[p.x, p.y, p.z] for traj in trajectories for p in traj.points])
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

    # Plot each trajectory with cycling colors
    for traj_id, traj in enumerate(trajectories):
        color = base_colors[traj_id % len(base_colors)]
        traj_pts = np.array([[p.x, p.y, p.z] for p in traj.points])

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
    points = load_points("3d_points.json")
    trajectories = detect_trajectories(points)
    for traj in trajectories:
        print(f"Trajectory {traj.id}: ({points.index(traj.points[0])}, {points.index(traj.points[-1])})")
    plot_trajectories(trajectories, plot_points=True, plot_markers=True)
