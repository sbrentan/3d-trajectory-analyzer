import json
import numpy as np
import matplotlib.pyplot as plt

def load_points(data_path):
    """Load 3D points from annotations, skipping nulls."""
    
    with open(data_path, "r") as f:
        data = json.load(f)

    if "annotations" not in data:
        return []
    
    pts = []
    for idx, ann in enumerate(data["annotations"]):
        p = ann.get("3d_point", None)
        if p is not None:
            pts.append((idx, np.array(p, dtype=float)))
    return pts

def detect_trajectories(points, min_length=3, threshold=0.001):
    """
    Detect trajectories by:
    1. Splitting at obstacles in X and Y
    2. Splitting at local minima in Z
    """
    if not points:
        return []

    xs = [p[1][0] for p in points]
    ys = [p[1][1] for p in points]
    zs = [p[1][2] for p in points]

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
    trajectories = []
    for start, end in zip(break_indices, break_indices[1:]):
        if end - start >= min_length:
            trajectories.append((start, end))

    return trajectories

def plot_trajectories(points, trajectories, plot_points=False, plot_markers=False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points faintly for context
    if plot_points:
        all_pts = np.array([p for _, p in points if p is not None])
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
    for t_id, (start, end) in enumerate(trajectories):
        color = base_colors[t_id % len(base_colors)]
        traj_pts = np.array([points[i][1] for i in range(start, end + 1) if points[i][1] is not None])

        ax.plot(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2],
                color=color, linewidth=2, alpha=0.9, label=f"Traj {t_id}")

        # Start & End markers
        if plot_markers:
            ax.scatter(traj_pts[0, 0], traj_pts[0, 1], traj_pts[0, 2], c="green", s=60, marker="o")
            ax.scatter(traj_pts[-1, 0], traj_pts[-1, 1], traj_pts[-1, 2], c="red", s=60, marker="x")

        # Add trajectory ID
        ax.text(traj_pts[0, 0], traj_pts[0, 1], traj_pts[0, 2] + 0.2, str(t_id),
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
    print("Detected trajectories (start_idx, end_idx):", trajectories)
    plot_trajectories(points, trajectories, plot_points=False, plot_markers=False)
