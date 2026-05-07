import open3d as o3d
import os

def debug_viewer(object_name, base_dataset_path, renders_dir):
    """
    Visualizes the Partial Point Cloud (Cyan), the Aligned Mesh (Grey), 
    and the Ground Truth (Red - optional) to verify completion and alignment.
    """
    # 1. Paths
    partial_path = os.path.join(base_dataset_path, "indata", f"{object_name}.ply")
    gt_path = os.path.join(base_dataset_path, "gtdata", f"{object_name}.ply")
    aligned_mesh_path = os.path.join(renders_dir, object_name, "mesh_bruteforce_aligned.ply")

    geometries = []

    # 2. Load and color the Aligned Mesh (The Result)
    # if os.path.exists(aligned_mesh_path):
    mesh = o3d.io.read_triangle_mesh(aligned_mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7]) # Grey "Clay"
    geometries.append(mesh)
    print(f"Loaded Aligned Mesh: {aligned_mesh_path}")
    # else:
    #     print(f"Warning: Aligned mesh not found at {aligned_mesh_path}")

    # 3. Load and color the Partial Point Cloud (The Input)
    # if os.path.exists(partial_path):
    pcd_partial = o3d.io.read_point_cloud(partial_path)
    pcd_partial.paint_uniform_color([0, 1, 1]) # Cyan
    geometries.append(pcd_partial)
    print(f"Loaded Partial PCD: {partial_path}")

    # 4. Load and color the Ground Truth (The Goal - Optional)
    # if os.path.exists(gt_path):
    pcd_gt = o3d.io.read_point_cloud(gt_path)
    pcd_gt.paint_uniform_color([1, 0, 0]) # Red
    # Note: We usually keep GT hidden or toggle it to see completion quality
    geometries.append(pcd_gt) 
    print(f"Loaded GT PCD: {gt_path}")

    # 5. Add a Coordinate Frame (X=Red, Y=Green, Z=Blue)
    # This helps you see if "Up" is actually Up.
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # 6. Launch Viewer
    print("\nControls:")
    print(" - [W] Wireframe toggle")
    print(" - [N] Normals toggle")
    print(" - [Q] Close viewer")
    
    o3d.visualization.draw_geometries(geometries, 
                                      window_name=f"SPAR3D Alignment Debug: {object_name}",
                                      width=1280, height=720,
                                      left=50, top=50,
                                      mesh_show_back_face=True)

if __name__ == "__main__":
    # Settings match your previous prompt
    # OBJ_NAME = "stanford-bunny"
    # OBJ_NAME = "horse"
    # OBJ_NAME = input("object: ")
      
    
    DATASET_BASE = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"
    RENDERS_DIR = "./renders"
    
    for OBJ_NAME in os.listdir("renders"):
        debug_viewer(OBJ_NAME, DATASET_BASE, RENDERS_DIR)