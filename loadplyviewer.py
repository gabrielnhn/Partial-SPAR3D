import trimesh
import os
import numpy as np


# object = "stanford-bunny"
object = "cow"
# object = "horse"

p1 = trimesh.load(f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/{object}.ply")
# # p2 = trimesh.load(f"./output/0/points.ply")
# p2 = trimesh.load(f"./renders/{object}/points.ply")

# # Create a rotation matrix for the final output domain
# rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
# rotation2 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
# output_rotation = rotation2 @ rotation
# # output_rotation = rotation

# # output_rotation = output_rotation.T

# p2 = p2.apply_transform(output_rotation)

# trimesh.Scene([p1, p2]).show()
scene = trimesh.Scene([p1])
# camera = trimesh.scene.Camera(fov=(30,30))
# camera = trimesh.scene.Camera(resolution=(1280, 720), fov=(30, 30))
scene.set_camera(fov=(60,60))
# scene.set_camera(fov=(30,30))
scene.show()

# # trimesh.Scene([p2]).show()

# mesh = trimesh.load(f"./renders/{object}/mesh.glb")
# mesh.show()