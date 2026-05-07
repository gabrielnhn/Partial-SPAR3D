import open3d as o3d
import numpy as np

# Set to your SPAR3D params
# SPAR3D_FOVY_DEG = 33.898
SPAR3D_FOVY_DEG = 60

def interactive_angle_finder(pcd_path):
    print(f"Loading {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Estimate normals for the "Clay" lighting to work in the live viewer
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Paint it grey
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    center = pcd.get_center()

    # Create interactive visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=512, height=512, window_name="SPAR3D POV Finder")

    # Match your material settings
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 1.0]) # Your blue background
    opt.point_size = 5.0 # Fat points
    opt.light_on = True

    vis.add_geometry(pcd)

    # Force the FOV to match SPAR3D
    ctr = vis.get_view_control()
    current_fov = ctr.get_field_of_view()
    ctr.change_field_of_view(step=(SPAR3D_FOVY_DEG - current_fov))

    # --- THE MAGIC REVERSE-MATH FUNCTION ---
    def capture_angle(vis):
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic

        # Extract Camera World Position from Extrinsic Matrix
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        eye = -R.T @ t

        # Reverse your rel_eye math to find the angles
        rel_eye = eye - center
        current_distance = np.linalg.norm(rel_eye)

        # e = arcsin(Y / dist)
        e_rad = np.arcsin(rel_eye[1] / current_distance)
        # a = arctan2(X, Z)
        a_rad = np.arctan2(rel_eye[0], rel_eye[2])

        e_deg = np.degrees(e_rad)
        a_deg = np.degrees(a_rad)

        print("\n" + "="*40)
        print("🎯 PERFECT ANGLE CAPTURED!")
        print(f"best_elev = {e_deg:.4f}")
        print(f"best_azim = {a_deg:.4f}")
        print(f"(Current Zoom Distance: {current_distance:.4f})")
        print("="*40 + "\n")
        return False

    # Bind the capture function to the 'P' key
    vis.register_key_callback(ord("P"), capture_angle)

    print("\n--- CONTROLS ---")
    print("🖱️ Left Click & Drag: Rotate object")
    print("🖱️ Scroll Wheel: Zoom in/out (adjusts distance)")
    print("🖱️ Right Click & Drag: Pan/Translate")
    print("⌨️ Press 'P': Print the current Azimuth and Elevation to the console")
    print("⌨️ Press 'Q': Quit")
    print("----------------\n")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Point this to one of your messed up objects
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata"
    object_file = "cow.ply" # or stanford-bunny.ply
    
    full_path = f"{dataset_path}/{object_file}"
    interactive_angle_finder(full_path)