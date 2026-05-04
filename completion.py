import torch
import numpy as np
import os
from time import time
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import open3d as o3d

RESOLUTION = 512

device = torch.device("cuda")

if not os.path.isdir("renders"):
    os.mkdir("renders")

import os
import sys
import gc
import torch
import numpy as np
from PIL import Image

device = "cuda"

def get_canonical_angles(pcd, pose_w=0.5, contour_w=0.2, area_w=1.0):
    W = H = 512
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    material = o3d.visualization.rendering.MaterialRecord()
    # material.shader = "defaultLit"
    material.shader = "defaultUnlit"
    material.base_color = [0.7, 0.7, 0.7, 1.0]
    material.point_size = 7.0
    renderer.scene.add_geometry("pcd", pcd, material)
    renderer.scene.set_background([0, 0, 0, 1])

    center = pcd.get_center()
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    # distance = np.linalg.norm(extent) * 1.5
    # distance = np.sqrt(((extent) ** 2).sum()) * 0.65
    distance = np.sqrt(((extent) ** 2).sum()) * 0.75

    # Conversion to torch for the point calculations (Chamfer/Pose)
    points_pt = torch.from_numpy(np.asarray(pcd.points)).float().cuda()

    startv, endv = -80.0, 80.0
    starth, endh = -180.0, 180.0
    num = 50 
    best_final_elev, best_final_azim = 0.0, 0.0

    for j in range(2):
        vers = torch.linspace(startv, endv, num)
        hors = torch.linspace(starth, endh, num)
        verss, horss = torch.meshgrid(vers, hors, indexing='ij')
        verss, horss = verss.flatten(), horss.flatten()
        
        best_loss = float('inf')
        
        for i in tqdm(range(len(verss)), desc="Finding canonical..."):
            elev_deg = verss[i].item()
            azim_deg = horss[i].item()
            
            e = np.radians(elev_deg)
            a = np.radians(azim_deg)
            
            rel_eye = np.array([
                distance * np.cos(e) * np.sin(a),
                distance * np.sin(e),
                distance * np.cos(e) * np.cos(a)
            ])
            eye = center + rel_eye
            
            renderer.setup_camera(60.0, center, eye, [0, 1, 0])
            
            # Render Depth and Image
            depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
            depth_np = np.asarray(depth_o3d)
            
            # Mask for valid points (Open3D depth is 0.0 for background usually)
            valid_mask = (depth_np > 0) & (np.isfinite(depth_np))
            
            
            if np.any(valid_mask):
                # Edge Score (Canny)
                channel = (valid_mask*255).astype(np.uint8)
                d_img = np.dstack((channel, channel, channel))
                
                edges = cv.Canny(d_img, 50, 150)
                contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                edge_score = len(contours)
                
                # Pose Regularization (Using Torch)
                eye_pt = torch.from_numpy(eye).float().cuda()
                posedist = torch.norm(eye_pt - points_pt, dim=-1).mean().item()
                
                valid_pixel_count = np.sum(valid_mask)
                # Normalize it against the total image area (W * H)
                area_score = valid_pixel_count / (W * H)

                loss = (pose_w * posedist) + (contour_w * edge_score) - (area_w * area_score)
                # print(f"posedist{posedist}, edge_score{edge_score}, area{area_score}")
                
                if loss < best_loss:
                    best_loss = loss
                    best_elev = elev_deg
                    best_azim = azim_deg
                    # best_depth = d_img
        
        # Hierarchical refinement
        interv = (endv - startv) / num
        interh = (endh - starth) / num
        startv, endv = best_elev - interv, best_elev + interv
        starth, endh = best_azim - interh, best_azim + interh
        best_final_elev, best_final_azim = best_elev, best_azim
        
    print(f"Optimal POV -> Azim: {best_final_azim:.1f}, Elev: {best_final_elev:.1f}")
    return best_final_elev, best_final_azim


import open3d as o3d
import numpy as np
import torch

def render_with_open3d(pcd, best_elev, best_azim, H=512, W=512):
    
    center = pcd.get_center()
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    # distance = np.linalg.norm(extent) * 1.5
    # distance = np.sqrt(((extent) ** 2).sum()) * 0.65
    distance = np.sqrt(((extent) ** 2).sum()) * 0.75
    
    e = np.radians(best_elev)
    a = np.radians(best_azim)
    
    # We calculate the eye position relative to the center
    rel_eye = np.array([
        distance * np.cos(e) * np.sin(a),
        distance * np.sin(e),
        distance * np.cos(e) * np.cos(a)
    ])
    
    eye = center + rel_eye
    
    # Orient normals so the "Clay" look works correctly
    pcd.orient_normals_towards_camera_location(camera_location=eye)
    
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    
    # "Clay" Material setup
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.7, 0.7, 0.7, 1.0]
    material.base_roughness = 0.5
    material.base_metallic = 0.0
    material.point_size = 5.0  # Adjust this float for thickness
    
    render.scene.add_geometry("pcd", pcd, material)
    render.scene.set_background([0, 0, 0, 1]) 
    
    # setup_camera(fov, center, eye, up)
    # In PyTorch3D look_at, the default 'up' is (0, 1, 0)
    render.setup_camera(60.0, center, eye, [0, 1, 0])
    
    image = render.render_to_image()
    depth = render.render_to_depth_image(z_in_view_space=True)
    
    return image, depth

def get_reference_image(pcd, best_elev, best_azim):
    print("Rendering PyTorch3D Reference Image and Depth...")
    img, depth = render_with_open3d(pcd, best_elev, best_azim)
    return img
    
from spar3d.system import SPAR3D
from transparent_background import Remover
from spar3d.utils import foreground_crop, remove_background
from contextlib import nullcontext
    
def spar3d_full(reference_images, reduction_count_type="keep", target_count=2000):

    model = SPAR3D.from_pretrained(
        "stabilityai/stable-point-aware-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=True,
    )
    model.to(device)
    model.eval()

    bg_remover = Remover(device=device)
    images = []
    idx = 0
    for image in reference_images:
        image = remove_background(
            Image.fromarray(
                np.asarray(image)
            ).convert("RGBA"), bg_remover
        )
        image = foreground_crop(image, crop_ratio=1.3)
        images.append(image)


    vertex_count = (
        -1
        if reduction_count_type == "keep"
        else (
            target_count
            if reduction_count_type == "vertex"
            else target_count // 2
        )
    )

    for i in tqdm(range(len(images))):
        image = images[i : i + 1]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.inference_mode():
            with (
                torch.autocast(device_type=device, dtype=torch.bfloat16)
                if "cuda" in device
                else nullcontext()
            ):
                mesh, glob_dict = model.run_image(
                    image,
                    bake_resolution=512,
                    remesh="none",
                    vertex_count=vertex_count,
                    return_points=True,
                )
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

        if len(image) == 1:
            out_mesh_path = os.path.join(renders_dir, "mesh.glb")
            mesh.export(out_mesh_path, include_normals=True)
            out_points_path = os.path.join(renders_dir, "points.ply")
            glob_dict["point_clouds"][0].export(out_points_path)
        else:
            for j in range(len(mesh)):
                out_mesh_path = os.path.join(renders_dir, f"mesh{str(i + j)}.glb")
                mesh[j].export(out_mesh_path, include_normals=True)
                out_points_path = os.path.join(renders_dir, f"points{str(i + j)}.ply")
                glob_dict["point_clouds"][j].export(out_points_path)


if __name__ == "__main__":
    print("----------")
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
    # object = "horse.ply"
    object = "stanford-bunny.ply"
    # object = "cow.ply"
    
    renders_dir = "renders"
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)    

    renders_dir = os.path.join(renders_dir, object.split(".")[0])
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)
    
    print("LOADING PCD;")
    partial_pcd = o3d.io.read_point_cloud(os.path.join(dataset_path, "indata", object))
    partial_pcd.estimate_normals()

    angles = {
        "stanford-bunny.ply": (12.016324043273926, -129.30612182617188),
        "cow.ply": (10.44897747039795,-2.3510241508483887),
    }
    
    # print("FIND AZIM/ELEV;")
    # if object in angles:
    #     best_elev, best_azim = angles[object]
    # else:
    best_elev, best_azim = get_canonical_angles(partial_pcd)
    
    # best_elev = 12.016324043273926
    # best_azim = -129.30612182617188

    print("GET BEST RGB;")
    canonical_image = get_reference_image(partial_pcd, best_elev, best_azim)
    o3d.io.write_image(os.path.join(renders_dir, "RENDER-pre.png"), canonical_image)
    
    spar3d_full([canonical_image])
    
