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

SPAR3D_FOVY_RAD = 0.591627
SPAR3D_FOVY_DEG = np.rad2deg(SPAR3D_FOVY_RAD)
# SPAR3D_FOVY_DEG = 60
SPAR3D_DISTANCE = 2.2


def get_canonical_angles(pcd, pose_w=0.5, contour_w=0.2, area_w=1.0):
    W = H = 512
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    material = o3d.visualization.rendering.MaterialRecord()
    # material.shader = "defaultLit"
    material.shader = "defaultUnlit"
    material.base_color = [0.7, 0.7, 0.7, 1.0]
    material.point_size = 5.0
    renderer.scene.add_geometry("pcd", pcd, material)
    renderer.scene.set_background([0, 0, 0, 1])

    center = pcd.get_center()
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    # distance = np.linalg.norm(extent) * 1.5
    # distance = np.sqrt(((extent) ** 2).sum()) * 0.65
    distance = np.sqrt(((extent) ** 2).sum()) * 0.8
    # distance = SPAR3D_DISTANCE

    # Conversion to torch for the point calculations (Chamfer/Pose)
    points_pt = torch.from_numpy(np.asarray(pcd.points)).float().cuda()

    startv, endv = -80.0, 80.0
    starth, endh = -180.0, 180.0
    num = 50 
    best_final_elev, best_final_azim = 0.0, 0.0
    best_elev, best_azim = 0.0, 0.0
    best_loss = float('inf')

    for j in range(2):
        vers = torch.linspace(startv, endv, num)
        hors = torch.linspace(starth, endh, num)
        verss, horss = torch.meshgrid(vers, hors, indexing='ij')
        verss, horss = verss.flatten(), horss.flatten()        
        
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
            # renderer.setup_camera(SPAR3D_FOVY_DEG, center, eye, [0, 1, 0])
            
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
                
                y_idx, x_idx = np.where(valid_mask)
                bbox_area = (x_idx.max() - x_idx.min() + 1) * (y_idx.max() - y_idx.min() + 1)
                
                # Normalize it against the total image area (W * H)
                # area_score = valid_pixel_count / (W * H)
                area_score = valid_pixel_count / bbox_area

                loss = (pose_w * posedist) + (contour_w * edge_score) - (area_w * area_score)
                # print(f"posedist{posedist}, edge_score{edge_score}, area{area_score}")
                
                if loss < best_loss:
                    best_loss = loss
                    best_elev = elev_deg
                    best_azim = azim_deg
                    best_depth = d_img
        
        # Hierarchical refinement
        interv = (endv - startv) / num
        interh = (endh - starth) / num
        startv, endv = best_elev - interv, best_elev + interv
        starth, endh = best_azim - interh, best_azim + interh
        best_final_elev, best_final_azim = best_elev, best_azim
        
    print(f"Optimal POV -> Azim: {best_final_azim:.1f}, Elev: {best_final_elev:.1f}")
    # print(f"Optimal POV -> Azim: {best_azim:.1f}, Elev: {best_elev:.1f}")
    cv.imwrite(
        os.path.join(renders_dir, "bestdepth.png"),
        best_depth
    )
    return best_final_elev, best_final_azim, distance
    # return best_elev, best_azim


import open3d as o3d
import numpy as np
import torch

def render_with_open3d(pcd, best_elev, best_azim, H=512, W=512):
    
    center = pcd.get_center()
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    # distance = np.linalg.norm(extent) * 1.5
    # distance = np.sqrt(((extent) ** 2).sum()) * 0.65
    distance = np.sqrt(((extent) ** 2).sum()) * 0.8
    # distance = SPAR3D_DISTANCE
    
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
    render.scene.set_background([0, 0, 1, 1]) 
    
    # In PyTorch3D look_at, the default 'up' is (0, 1, 0)
    render.setup_camera(60.0, center, eye, [0, 1, 0])
    # render.setup_camera(SPAR3D_FOVY_DEG, center, eye, [0, 1, 0])
    
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
    
def spar3d_full(reference_images,
                objects,
                reduction_count_type="keep",
                target_count=2000):

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
                    # NEW STUFF
                    # custom_distance=distances[i],
                    # custom_fovy_deg=60,
                )
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

        renders_dir = os.path.join("renders", os.path.basename(objects[i]).split(".")[0])   
        out_mesh_path = os.path.join(renders_dir, "mesh.glb")
        mesh.export(out_mesh_path, include_normals=True)
        out_points_path = os.path.join(renders_dir, "points.ply")
        glob_dict["point_clouds"][0].export(out_points_path)

    del model, bg_remover




from scipy.spatial import KDTree
import itertools
import copy

# taken from SPAR3D
def brute_force_align_and_eval(mesh_path, gt_pcd_path, num_samples=16384, d_th=0.05):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
    
    # Sample dense points from mesh
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=num_samples)

    def normalize_pcd(pcd):
        pcd_copy = copy.deepcopy(pcd)
        center = pcd_copy.get_center()
        pcd_copy.translate(-center)
        bbox = pcd_copy.get_axis_aligned_bounding_box()
        max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())
        pcd_copy.scale(1.0 / max_extent, center=(0, 0, 0))
        return pcd_copy, center, max_extent

    pred_norm, _, _ = normalize_pcd(mesh_pcd)
    gt_norm, gt_center, gt_scale = normalize_pcd(gt_pcd)

    # print("brute-force search over SO(3) [13,824 rotations]...")
    
    # Downsample for extreme speed during the brute-force phase (1000 points is plenty for rough alignment)
    src_down = pred_norm.random_down_sample(1000 / len(pred_norm.points))
    tgt_down = gt_norm.random_down_sample(1000 / len(gt_norm.points))
    
    src_pts = np.asarray(src_down.points)
    tgt_pts = np.asarray(tgt_down.points)
    
    # Build KDTree on the Ground Truth for lightning-fast distance queries
    tgt_tree = KDTree(tgt_pts)
    
    best_dist = float('inf')
    best_R = np.eye(3)
    
    # Search grid: Every 15 degrees around X, Y, and Z
    angles = np.deg2rad(np.arange(0, 360, 15))
    
    for rx, ry, rz in itertools.product(angles, angles, angles):
        # Fast manual rotation matrix construction
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        # Apply rotation to source points
        rot_src = src_pts @ R.T
        
        # Calculate mean distance to nearest neighbors
        d, _ = tgt_tree.query(rot_src)
        dist = np.mean(d)
        
        if dist < best_dist:
            best_dist = dist
            best_R = R

    print(f"   -> Found optimal initial rotation.")

    # Apply the best rotation to our normalized dense prediction
    pred_norm.rotate(best_R, center=(0,0,0))

    print("Refining with ICP")
    threshold = 0.05
    trans_init = np.eye(4)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pred_norm, gt_norm, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
    )
    
    pred_norm.transform(reg_p2p.transformation)

    # 5. Restore Original Ground Truth Scale & Position (Optional but useful for viewing)
    pred_norm.scale(gt_scale, center=(0,0,0))
    pred_norm.translate(gt_center)
    
    mesh.translate(-mesh.get_center())
    mesh.scale(1.0 / np.max((mesh.get_axis_aligned_bounding_box().get_max_bound() - mesh.get_axis_aligned_bounding_box().get_min_bound())), center=(0,0,0))
    mesh.rotate(best_R, center=(0,0,0))
    mesh.transform(reg_p2p.transformation)
    mesh.scale(gt_scale, center=(0,0,0))
    mesh.translate(gt_center)
    
    aligned_path = mesh_path.replace(".glb", "_bruteforce_aligned.ply")
    o3d.io.write_triangle_mesh(aligned_path, mesh)
    
    dists_m_to_gt = np.asarray(pred_norm.compute_point_cloud_distance(gt_pcd))
    dists_gt_to_m = np.asarray(gt_pcd.compute_point_cloud_distance(pred_norm))
    
    mean_dist_m_to_gt = np.mean(dists_m_to_gt)
    mean_dist_gt_to_m = np.mean(dists_gt_to_m)
    
    chamfer_dist = mean_dist_m_to_gt + mean_dist_gt_to_m
    
    # precision = np.mean(dists_m_to_gt < d_th) * 100
    # recall = np.mean(dists_gt_to_m < d_th) * 100    
    # f_score = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    print(f"\n--- SPAR3D BENCHMARK RESULTS ---")
    print(f"Chamfer Distance:    {chamfer_dist:.6f}")
    # print(f"Accuracy (M->GT):    {mean_dist_m_to_gt:.6f}")
    # print(f"Completeness (GT->M):{mean_dist_gt_to_m:.6f}")
    # print(f"F-Score @ {d_th}:      {f_score:.2f}%")
    # print(f"--------------------------------\n")

    return chamfer_dist


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None)
parser.add_argument("--dir", default=None)
parser.add_argument("--fov", type=float, default=SPAR3D_FOVY_DEG)
parser.add_argument("--distance", type=float, default=SPAR3D_DISTANCE)
args = parser.parse_args()
MYCHOICEFOV = args.fov

if __name__ == "__main__":
    print("----------")
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    

    # object = "horse.ply"
    # object = "stanford-bunny.ply"
    # object = "cow.ply"
    
    objects = []
    if not args.file and not args.dir:
        # object = os.path.join(dataset_path, "indata", input("object: ")+".ply")
        object = os.path.join(dataset_path, "gtdata", input("object: ")+".ply")
        objects.append(object)

    else:
        if os.path.isdir(args.dir):
            for file in os.listdir(args.dir):
                if file.endswith(".ply"):
                    objects.append(
                        os.path.join(args.dir, file)
                    )
                
        
        elif os.path.exists(args.file):
            objects.append(args.file)

        else:
            print("bruh give me a point cloud kom op")

    reference_images = []
    for object in tqdm(objects, desc=f"Processing all objects in {args.dir}"):
        print(f"PROCESSING {object}")
        
        renders_dir = "renders"
        if not os.path.isdir(renders_dir):
            os.mkdir(renders_dir)    

        renders_dir = os.path.join(renders_dir, os.path.basename(object).split(".")[0])
        if not os.path.isdir(renders_dir):
            os.mkdir(renders_dir)
        
        print("LOADING PCD;")
        # partial_pcd = o3d.io.read_point_cloud()
        partial_pcd = o3d.io.read_point_cloud(object)
        if not partial_pcd:
            print("SOMETHING WENT TERRIBLY WRONG DUDE")
            exit()
        print("RIGHT AFTER")
        
        partial_pcd.estimate_normals()        
        best_elev, best_azim, distance = get_canonical_angles(partial_pcd)

        print("GET BEST RGB;")
        canonical_image = get_reference_image(partial_pcd, best_elev, best_azim)
        o3d.io.write_image(os.path.join(renders_dir, "RENDER-pre.png"), canonical_image)
        reference_images.append(canonical_image)

    # exit()
    spar3d_full(reference_images, objects)
        
        
    for object in objects:
        renders_dir = os.path.join("renders", os.path.basename(object).split(".")[0])   
        raw_mesh_path = os.path.join(renders_dir, "mesh.glb")
        gt_path = object.replace("indata", "gtdata")
        
        print(f"stats for {os.path.basename(object)}")
        brute_force_align_and_eval(
            mesh_path=raw_mesh_path,
            gt_pcd_path=gt_path,
            num_samples=10000, 
            d_th=0.05 
        )