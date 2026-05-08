import open3d as o3d
import os

def debug_viewer(object_name, base_dataset_path, renders_dir):
    """
    Visualizes the Partial Point Cloud (Cyan), the Aligned Mesh (Grey), 
    and the Ground Truth (Red - optional) to verify completion and alignment.
    """
    partial_path = os.path.join(base_dataset_path, "indata", f"{object_name}.ply")
    gt_path = os.path.join(base_dataset_path, "gtdata", f"{object_name}.ply")
    aligned_mesh_path = os.path.join(renders_dir, object_name, "mesh_bruteforce_aligned.ply")

    geometries = []

    mesh = o3d.io.read_triangle_mesh(aligned_mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7]) 
    geometries.append(mesh)
    print(f"Loaded Aligned Mesh: {aligned_mesh_path}")

    pcd_partial = o3d.io.read_point_cloud(partial_path)
    pcd_partial.paint_uniform_color([0, 1, 1])
    geometries.append(pcd_partial)
    print(f"Loaded Partial PCD: {partial_path}")

    # pcd_gt = o3d.io.read_point_cloud(gt_path)
    # pcd_gt.paint_uniform_color([1, 0, 0])
    # geometries.append(pcd_gt) 
    # print(f"Loaded GT PCD: {gt_path}")

    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # geometries.append(coord_frame)

    o3d.visualization.draw_geometries(geometries, 
                                      window_name=f"SPAR3D Alignment Debug: {object_name}",
                                      width=1280, height=720,
                                      left=50, top=50,
                                      mesh_show_back_face=True)

if __name__ == "__main__":
    # OBJ_NAME = "stanford-bunny"
    # OBJ_NAME = "horse"
    # OBJ_NAME = input("object: ")
      
    
    # DATASET_BASE = "/home/gabrielnhn/datasets/synthetic_redwood/upload/redwood"
    DATASET_BASE = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"
    RENDERS_DIR = "./renders"
    
    for OBJ_NAME in os.listdir(DATASET_BASE+"/indata"):
        OBJ_NAME=OBJ_NAME.replace(".ply", "")
        debug_viewer(OBJ_NAME, DATASET_BASE, RENDERS_DIR)