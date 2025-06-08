import sys
import numpy as np
import open3d as o3d
import torch

def np_to_o3d(array: np.ndarray):
    assert (shape := array.shape[-1]) in (3, 6)
    pos = array[:, :3]
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pos))
    if shape == 6:
        color = array[:, 3:]
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def pca_axes(pcd):
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh.astype(np.float32)  # shape (3, 3)

    return components, centered

def disambiguate(centered_pcd, eig_vecs):
    """
    Function takes in centered_pcd and returns PCA normalized cloud
    """

    #if cos(theta) is +, then point is on one side of centroid, and vice versa

    for i in range(3):
        pos_sum, neg_sum = line_distance_sums(centered_pcd, eig_vecs[i])

        if neg_sum > pos_sum:
        # invert direction
            eig_vecs[i] *= -1


    return centered_pcd @ eig_vecs.T



def line_distance_sums(centered_points: torch.Tensor,
                       direction: torch.Tensor):
    """
    points      : (N,3) centered cloud (float32/64, CPU or CUDA)
    direction   : (3,)  line direction (need not be unit length)

    Returns
    -------
    pos_sum, neg_sum  (each scalar tensor)
    """

    v_hat = direction / direction.norm()          # (3,)
    dots  = centered_points @ v_hat                        # (N,)  u·v  (torch.mv is fine too)

    # squared distance ‖u‖² - (u·v̂)²
    r2 = centered_points.pow(2).sum(dim=1) - dots.pow(2)   # (N,)

    dists = r2.clamp_min_(0.)                 # avoid tiny neg. due to FP error

    # masks for the two half‑spaces
    pos_mask = dots > 0
    neg_mask = dots < 0

    pos_sum = dists[pos_mask].sum()
    neg_sum = dists[neg_mask].sum()
    return pos_sum, neg_sum

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python view_pcd_with_pca.py <cloud.ply>")
        sys.exit(1)

    cloud_path = sys.argv[1]
    pcd = o3d.io.read_point_cloud(cloud_path)


    components, centered = pca_axes(pcd) #ROWS ARE THE BASIS

    pcd = disambiguate(torch.tensor(centered).float(), torch.tensor(components).float())

    pcd = np_to_o3d(pcd.numpy())

    o3d.visualization.draw_geometries([pcd])
