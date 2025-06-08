import sys
import numpy as np
from pprl.utils.o3d import np_to_o3d, o3d_to_np
import open3d as o3d


if len(sys.argv) != 2:
    print("Usage: python view_pcd.py <pointcloud.ply>")
    sys.exit(1)

pcd = o3d.io.read_point_cloud(sys.argv[1])


# camera = np.array([0, -175, 120])
#
# diameter = np.linalg.norm(
#     np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
# )
#
# radius = diameter * 100
#
# _, pt_map = pcd.hidden_point_removal(camera, radius)
#
# pcd = pcd.select_by_index(pt_map)

o3d.visualization.draw_geometries([pcd])
