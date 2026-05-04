import trimesh
import os
import numpy as np


# object = "cow"
object = "stanford-bunny"
# object = "horse"

p1 = trimesh.load(f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/{object}.ply")
# # p2 = trimesh.load(f"./output/0/points.ply")
mesh = trimesh.load(f"./renders/{object}/mesh.glb")
# p2 = trimesh.load(f"./renders/{object}/points.ply")
p2 = mesh



# # Create a rotation matrix for the final output domain
rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
rotation2 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
output_rotation = rotation2 @ rotation
# output_rotation = rotation

# output_rotation = output_rotation.T

p2 = p2.apply_transform(output_rotation)

scene = trimesh.Scene([p1, p2])
# scene = trimesh.Scene([p1])
# scene = trimesh.Scene([p2])


scene.set_camera(fov=(60,60))
# scene.set_camera(fov=(30,30))
scene.show()

# # trimesh.Scene([p2]).show()

# mesh.show()