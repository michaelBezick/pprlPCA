import numpy as np
import open3d as o3d

# ---------- big‑radius HPR as requested ---------------------------------
def hpr_partial(points: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """
    Same heuristic as your standalone script:
    radius = cloud diameter * 100
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    diameter = np.linalg.norm(points.max(0) - points.min(0))
    radius   = diameter * 100.0
    _, idx = pcd.hidden_point_removal(eye, radius)
    return points[np.asarray(idx)]

# ---------- eye samplers -------------------------------------------------
def random_eye(radius: float = 200.0,
               lookat: np.ndarray = np.array([10., 0., 55.]),
               z_floor: float = 0.0) -> np.ndarray:
    theta = np.random.uniform(-np.pi,  np.pi)
    phi   = np.random.uniform(0, np.pi/2)         # upper hemisphere
    x = radius*np.cos(theta)*np.sin(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = max(z_floor, radius*np.cos(phi))
    return lookat + np.array([x, y, z], np.float32)
