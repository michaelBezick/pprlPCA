#!/usr/bin/env python3
"""
view_pcd_with_pca.py  <point_cloud.ply>  [scale_factor]

• Shows the point cloud plus its 3 PCA axes.
• `scale_factor` (optional, default = 1.0) controls how long the axes are
  relative to the cloud’s bounding‑box diagonal.
    – 1.0  → axes ~ the size of the cloud
    – 2.0  → twice as long, etc.
"""

import sys
import numpy as np
import open3d as o3d


def pca_axes(pcd, scale_factor=1.0):
    """Return a LineSet of three PCA axes scaled by `scale_factor`."""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    eig_vecs = eig_vecs[:, eig_vals.argsort()[::-1]]  # sort big→small

    # Length reference: diagonal of the AABB (overall cloud size)
    diag = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    length = scale_factor * diag

    # Build one shared start point (centroid) + 3 end points
    points = [centroid]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB
    for i in range(3):
        points.append(centroid + eig_vecs[:, i] * length)

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]),
    )
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset


def main(path, scale_factor):
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        sys.exit("❌ Empty or unreadable point cloud.")
    axes = pca_axes(pcd, scale_factor)

    o3d.visualization.draw_geometries(
        [pcd, axes],
        window_name=f"PCA axes (scale × {scale_factor})",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python view_pcd_with_pca.py <cloud.ply> [scale_factor]")
        sys.exit(1)

    cloud_path = sys.argv[1]
    sf = float(sys.argv[2]) if len(sys.argv) == 3 else 1.0
    main(cloud_path, sf)

