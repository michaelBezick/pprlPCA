import numpy as np
import open3d as o3d

# ---------- big‑radius HPR as requested ---------------------------------
def hpr_partial(points: np.ndarray, eye: np.ndarray, min_pts=100) -> np.ndarray:
    """
    Same heuristic as your standalone script:
    radius = cloud diameter * 100
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    diameter = np.linalg.norm(points.max(0) - points.min(0))
    radius   = diameter * 2000
    _, idx = pcd.hidden_point_removal(eye, radius)

    idx = np.asarray(idx)

    if idx.size < min_pts:
        return points


    return points[idx]

# ---------- eye samplers -------------------------------------------------
def random_eye(radius: float = 200.0,
               lookat: np.ndarray = np.array([10., 0., 55.]),
               z_floor: float = 70) -> np.ndarray:
    theta = np.random.uniform(-np.pi,  np.pi)
    phi   = np.random.uniform(0, np.pi/2)         # upper hemisphere
    x = radius*np.cos(theta)*np.sin(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = max(z_floor, radius*np.cos(phi))
    return lookat + np.array([x, y, z], np.float32)

def perfect_eye():
    lookat = np.array([10., 0., 55.])
    loc = np.array([0., 0., 200.])
    return lookat + loc

class EpisodeHPR:
    """
    Callable that applies hidden‑point‑removal with ONE eye per episode.

    Usage
    -----
    hpr = EpisodeHPR(eye_sampler=random_eye)     # create once
    pts  = hpr(pts)                              # called every step
    hpr.new_episode()                            # call inside env.reset()
    """
    def __init__(self, eye_sampler):
        self.eye_sampler = eye_sampler
        self.eye: np.ndarray | None = None      # set on first call

    # -------- interface the post‑processing hook expects --------
    def __call__(self, points: np.ndarray) -> np.ndarray:
        if self.eye is None:                     # first step of episode
            self.eye = self.eye_sampler()
        return hpr_partial(points, self.eye)

    # -------- tell wrapper that a new episode starts ------------
    def new_episode(self):
        self.eye = None
