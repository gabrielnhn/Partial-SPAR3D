import trimesh
import os
import numpy as np

p1 = trimesh.load(f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/stanford-bunny.ply")
# p2 = trimesh.load(f"./output/0/points.ply")
p2 = trimesh.load(f"./renders/stanford-bunny/points.ply")

# Create a rotation matrix for the final output domain
rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
rotation2 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
output_rotation = rotation2 @ rotation
# output_rotation = rotation

# output_rotation = output_rotation.T

p2 = p2.apply_transform(output_rotation)

trimesh.Scene([p1, p2]).show()
# trimesh.Scene([p2]).show()
