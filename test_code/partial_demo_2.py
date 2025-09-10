#!/usr/bin/env python3
"""
partial_from_complete.py   complete_cloud.ply   [n_views]

• Loads a full point cloud (e.g. a fused LiDAR scan).
• Generates `n_views` realistic partial clouds with
  hidden‑point‑removal (HPR) – no meshes required.
• Opens TWO Open3D viewers so the clouds don’t overlap.
• Prints timing for each step.
"""

import sys, time, numpy as np, open3d as o3d, random

AXIS_SCALE = 1.2              # length of PCA axes (× diagonal)
N_VIEWS    = int(sys.argv[2]) if len(sys.argv) == 3 else 1

# ------------------------------------------------------------------ helpers ---

def pca_axes(pcd, scale=1.0):
    pts   = np.asarray(pcd.points)
    mean  = pts.mean(0)
    cov   = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eigh(cov)
    vecs  = vecs[:, vals.argsort()[::-1]]
    diag  = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    L     = diag * scale
    pts_ax= [mean] + [mean + vecs[:, i]*L for i in range(3)]
    lines = [[0,1],[0,2],[0,3]]
    colors= [[1,0,0],[0,1,0],[0,0,1]]
    ls    = o3d.geometry.LineSet(
               points=o3d.utility.Vector3dVector(pts_ax),
               lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def show(pcd, title, colour, scale=1.0):
    o3d.visualization.draw_geometries(
        [pcd.paint_uniform_color(colour), pca_axes(pcd, scale)],
        window_name=title, width=1280, height=720)

def random_camera(bbox, rad_mult=1.2):
    c   = bbox.get_center()
    r   = np.linalg.norm(bbox.get_extent())/2 * rad_mult
    d   = np.random.randn(3); d /= np.linalg.norm(d)
    return c + d*r, r*1.05                   # cam‑pos, finite radius

def show_window(name, geom, x, y):
    app  = o3d.visualization.gui.Application.instance
    app.initialize()
    win  = app.create_window(name, 800, 600, x, y)
    scn  = o3d.visualization.rendering.Open3DScene(win.renderer)
    scn.add_geometry(name, geom, o3d.visualization.rendering.MaterialRecord())
    bbox= geom.get_axis_aligned_bounding_box()
    scn.setup_camera(60, bbox, bbox.get_center())
    return win


def partial(full):
    cam, radius = random_camera(full.get_axis_aligned_bounding_box())
    _, idx = full.hidden_point_removal(cam, np.linalg.norm(cam))

    return full.select_by_index(idx)

full = o3d.io.read_point_cloud(sys.argv[1])
if full.is_empty(): sys.exit("❌ empty cloud")

partials = [partial(full) for _ in range(N_VIEWS)]
clouds = [full, *partials]
titles   = ["COMPLETE cloud", *[f"PARTIAL #{i}" for i in range(len(partials))]]

# ----- single Visualizer, space‑bar to go to next cloud ----------------------
vis   = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name=titles[0], width=1280, height=720)
idx   = 0

def next_cloud(vis):
    global idx
    idx = (idx + 1) % len(clouds)
    vis.clear_geometries()
    vis.add_geometry(clouds[idx])
    vis.get_render_option().point_size = 3.0
    vis.update_renderer()
    vis.poll_events()
    vis.get_view_control().set_zoom(0.8)
    print(f"Showing {titles[idx]}")
    return False                      # keep running

vis.register_key_callback(ord(" "), next_cloud)    # space bar
vis.add_geometry(clouds[0])
print("Press SPACE to cycle through views, ESC to quit.")
vis.run()
vis.destroy_window()


# show(full, "COMPLETE cloud + PCA", [0.6]*3, AXIS_SCALE)   # viewer 1
#
# for v in range(N_VIEWS):
#     cam, radius = random_camera(full.get_axis_aligned_bounding_box())
#     t0 = time.time()
#     _, idx   = full.hidden_point_removal(cam, radius)
#     partial  = full.select_by_index(idx)
#     print(f"[view {v}] partial size {len(partial.points):,d} "
#           f"(HPR { (time.time()-t0)*1000:.1f} ms)")
#     # show(partial, f"PARTIAL cloud #{v}", [1,0.6,0.3], AXIS_SCALE*0.8)  # viewer 2
#     wins.append(show_window(f"PARTIAL #{v}", partial, 30+820*v, 40))
#     # o3d.io.write_point_cloud("partial.ply")
#
# o3d.visualization.gui.Application.instance.run()
