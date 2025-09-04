import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# Load point clouds from .ply files
pcd_camera = o3d.io.read_point_cloud("camera_frame.ply")
pcd_world = o3d.io.read_point_cloud("world_frame.ply")
pcd_pca = o3d.io.read_point_cloud("pca_frame.ply")

# Convert to NumPy arrays
points_camera = np.asarray(pcd_camera.points)
points_world = np.asarray(pcd_world.points)
points_pca = np.asarray(pcd_pca.points)

# List for easy looping
point_clouds = [points_camera, points_world, points_pca]
titles = ["Camera Frame", "World Frame", "PCA Frame"]

# Create side-by-side 3D plots
fig = plt.figure(figsize=(15, 5))

for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.scatter(point_clouds[i][:, 0], point_clouds[i][:, 1], point_clouds[i][:, 2])
    ax.set_title(titles[i], fontsize=30)
    ax.view_init(elev=20, azim=-60)  # Default-ish 3D view

plt.tight_layout(pad=2.)
plt.savefig("frames_comparison.pdf", format="pdf")
plt.show()

