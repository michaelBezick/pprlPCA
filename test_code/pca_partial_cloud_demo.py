#!/usr/bin/env python3

"""
pca_partial_cloud_demo.py

• Generates a synthetic “complete” point cloud from several depth renders
• Extracts a realistic partial view via occlusion‑aware ray‑casting
• Plots both clouds with their 3 PCA axes
• Prints wall‑clock timings
"""

import time
import numpy as np
import open3d as o3d
from pathlib import Path
import argparse
import random
import open3d.core as o3c

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
N_VIEWS = 20         # depth images to fuse for the complete cloud
IMG_RES = 256         # render resolution
PARTIAL_FOV = 60      # degrees
PARTIAL_RAD = 1.1     # camera distance multiplier
AXIS_SCALE = 1.2      # axes length relative to bbox diagonal
USE_GPU   = False     # set True if you have PyTorch3D + CUDA

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def mesh_to_depth(mesh, cam_pos, fov_deg=60, res=256):
    """
    Off‑screen CPU ray‑cast:
        • cam_pos  – numpy (3,) world coords
        • returns  – Open3D PointCloud of hit points
    """
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    center = o3c.Tensor([0.0, 0.0, 0.0], dtype=o3c.Dtype.Float32)          # look‑at target
    eye    = o3c.Tensor(cam_pos.astype(np.float32), dtype=o3c.Dtype.Float32)
    up     = o3c.Tensor([0.0, 1.0, 0.0], dtype=o3c.Dtype.Float32)

    rays   = scene.create_rays_pinhole(                                   # new signature
        fov_deg, center, eye, up, res, res
    )                                                                      # (H*W, 6)

    ans    = scene.cast_rays(rays)                                         # dict of tensors
    t_hit  = ans['t_hit'].numpy()                                          # (N,)
    rays   = rays.numpy()                                                  # (N, 6)

    mask   = np.isfinite(t_hit)
    hits   = rays[mask, :3] + rays[mask, 3:] * t_hit[mask][:, None]

    pcd    = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hits)
    return pcd
def look_at(cam_pos, target=np.zeros(3), up=np.array([0, 1, 0])):
    """
    Build a 3×3 rotation matrix that points the camera located at `cam_pos`
    toward `target` with the given `up` vector.
    """
    z = (target - cam_pos)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

def load_mesh(path=None):
    """Return an Open3D TriangleMesh.
       • if `path` is given, load that file;
       • otherwise download the Stanford bunny demo mesh."""
    if path is None:
        bunny = o3d.data.BunnyMesh()          # downloads if not cached
        mesh_path = getattr(bunny, "path", None) or bunny.mesh_path
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    else:
        mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()
    return mesh



def fuse_depth_views(mesh, n_views=N_VIEWS, res=IMG_RES):
    """Create a dense cloud by orbiting cameras on a sphere."""
    t0 = time.time()
    cloud = o3d.geometry.PointCloud()

    # sample camera centres on a sphere (icosphere vertices)
    ico = o3d.geometry.TriangleMesh.create_sphere(1.0).subdivide_midpoint(1)
    cams = np.asarray(ico.vertices)[:n_views]

    for c in cams:
        cam_pos = c * 1.3                        # just outside the unit mesh
        # R = look_at(cam_pos)                    # <-- fixed
        pcd = mesh_to_depth(mesh, cam_pos, res=res)
        cloud += pcd

    cloud = cloud.voxel_down_sample(0.002)
    print(f"complete cloud: {len(cloud.points):,d} pts, "
          f"{(time.time()-t0)*1000:.1f} ms")
    return cloud

def hidden_point_removal_partial(cloud, rad=PARTIAL_RAD):
    """
    Make a partial cloud that mimics a single‑frame depth capture
    by keeping only the points visible from a random camera pose.

    • rad  – multiplier of the object’s bounding‑sphere radius
             (1.1 → camera is 10 % beyond the object)
    """
    t0 = time.time()
    bbox    = cloud.get_axis_aligned_bounding_box()
    center  = bbox.get_center()
    bsphere = np.linalg.norm(bbox.get_extent()) / 2          # radius of a sphere
                                                            # that encloses the AABB
    # random direction on the unit sphere
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    cam = center + direction * bsphere * rad                 # camera position
    radius = np.linalg.norm(center - cam) + bsphere * 1.05   # finite search radius

    _, idx = cloud.hidden_point_removal(cam, radius)         # ← finite radius!
    partial = cloud.select_by_index(idx)

    print(f"partial cloud:  {len(partial.points):,d} pts, "
          f"{(time.time()-t0)*1000:.1f} ms")
    return partial

def pca_axes(pcd, scale=1.0):
    pts = np.asarray(pcd.points)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, vals.argsort()[::-1]]
    diag = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    L = diag * scale
    points = [mean]
    for i in range(3):
        points.append(mean + vecs[:, i] * L)
    lines = [[0,1],[0,2],[0,3]]
    colors = [[1,0,0],[0,1,0],[0,0,1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def show_with_axes(pcd, title, axis_scale=1.0, color=[0.6, 0.6, 0.6]):
    objs = [
        pcd.paint_uniform_color(color),
        pca_axes(pcd, axis_scale),
    ]
    o3d.visualization.draw_geometries(
        objs, window_name=title, width=1280, height=720
    )

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(mesh_path=None):
    mesh = load_mesh(mesh_path)
    complete = fuse_depth_views(mesh)
    partial  = hidden_point_removal_partial(complete)

    o3d.io.write_point_cloud("complete.ply", complete)
    exit()

    show_with_axes(
        partial,  "PARTIAL cloud (orange) with PCA axes", 0.8 * AXIS_SCALE, [1, 0.6, 0.3]
    )

    show_with_axes(
        complete, "COMPLETE cloud (grey) with PCA axes", AXIS_SCALE, [0.6, 0.6, 0.6]
    )


    # vis_objects = [
    #     complete.paint_uniform_color([0.6,0.6,0.6]),
    #     pca_axes(complete, AXIS_SCALE),
    #     partial.paint_uniform_color([1,0.6,0.3]),
    #     pca_axes(partial, AXIS_SCALE*0.8)
    # ]
    # o3d.visualization.draw_geometries(
    #     vis_objects,
    #     window_name="Complete (grey)   +   Partial (orange)   with PCA axes",
    #     width=1280, height=720)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", nargs="?", help="mesh file to use (.stl/.ply/…)")
    args = parser.parse_args()
    main(args.mesh)

