

# object = "cow"
object = "stanford-bunny"
# object = "horse"
# object = "teapot"

import open3d as o3d
drag = "../datasets/synthetic_redwood/upload/plyobj/indata/xyzrgb_dragon.ply"
dragon = o3d.io.read_point_cloud(drag)
geometries = [dragon]
o3d.visualization.draw_geometries(geometries)

exit()

import trimesh
import os
import numpy as np


p1 = trimesh.load(f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/{object}.ply")
# # p2 = trimesh.load(f"./output/0/points.ply")
mesh = trimesh.load(f"./renders/{object}/mesh.glb")
mesh_aligned = trimesh.load(f"./renders/{object}/mesh_aligned.ply")
# p2 = trimesh.load(f"./renders/{object}/points.ply")
# p2 = mesh



# # Create a rotation matrix for the final output domain
# rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
# rotation2 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
# output_rotation = rotation2 @ rotation
# output_rotation = rotation

# # output_rotation = output_rotation.T
# mesh_aligned = mesh_aligned.apply_transform(output_rotation)


# p2 = p2.apply_transform(output_rotation)

# scene = trimesh.Scene([p1, mesh, mesh_aligned])
# scene = trimesh.Scene([p1,mesh_aligned])
scene = trimesh.Scene([mesh,mesh_aligned])
# scene = trimesh.Scene([p1])
# scene = trimesh.Scene([p2])


scene.set_camera(fov=(60,60))
# scene.set_camera(fov=(30,30))
scene.show()

# # trimesh.Scene([p2]).show()

# mesh.show()