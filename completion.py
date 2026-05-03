import torch
import numpy as np
import os
from time import time
from tqdm import tqdm
import cv2 as cv
from PIL import Image

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
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)


# def find_best_reference_pov_full(pcd, pose_w=0.5, edge_w=0.1):
#     """
#     Combines COMPC (Chamfer Distance + Depth Regularization) 
#     with OpenCV Depth-Edge Contour detection.
#     """
#     points = pcd.points_padded()[0]
#     total_points = points.shape[0]
    
#     # Calculate scene bounds for camera placement
#     bbox = pcd.get_bounding_boxes()
#     bbox_min = bbox.min(dim=-1).values[0]
#     bbox_max = bbox.max(dim=-1).values[0]
#     bbox_center = (bbox_min + bbox_max) / 2.0
#     distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65
#     # distance = 4.0

#     image_size = 512
#     raster_settings = PointsRasterizationSettings(
#         image_size=image_size, 
#         radius=0.01, 
#         points_per_pixel=1,
#         bin_size=0
#     )

#     startv, endv = -80.0, 80.0
#     starth, endh = -180.0, 180.0
#     num = 50 
#     # num = 20 
#     # num = 40 
#     batch_size = 16
#     best_final_elev, best_final_azim = 0.0, 0.0

#     for j in range(2):
#         vers = torch.linspace(startv, endv, num, device=device)
#         hors = torch.linspace(starth, endh, num, device=device)
#         verss, horss = torch.meshgrid(vers, hors, indexing='ij')
#         verss, horss = verss.flatten(), horss.flatten()
        
#         best_loss = float('inf')
#         best_elev, best_azim = 0.0, 0.0
#         best_img_to_save = None
        
#         for i in range(0, len(verss), batch_size):
#             chunk_elevs = verss[i:i+batch_size]
#             chunk_azims = horss[i:i+batch_size]
            
#             R, T = look_at_view_transform(dist=distance, elev=chunk_elevs, azim=chunk_azims, device=device, at=bbox_center.unsqueeze(0))
#             cameras = PerspectiveCameras(device=device, R=R, T=T)
#             # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30.0)
            
#             rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            
#             pcd_batch = pcd.extend(len(chunk_elevs))
#             fragments = rasterizer(pcd_batch)
            
#             depth_maps = fragments.zbuf[..., 0] 
#             idx_map = fragments.idx[..., 0] 
#             cam_centers = cameras.get_camera_center() 
            
#             for b in range(len(chunk_elevs)):
#                 valid_mask = idx_map[b] != -1
#                 visible_indices = torch.unique(idx_map[b][valid_mask])
                
#                 if len(visible_indices) > 0:
#                     visible_indices = visible_indices % total_points
#                     visible_pts = points[visible_indices] 
                    
#                     # Chamfer-like fidelity
#                     dists, _, _ = knn_points(points.unsqueeze(0), visible_pts.unsqueeze(0), K=1)
#                     fixed_cd = dists.squeeze().sqrt().mean() 
                    
#                     # Pose distance regularization
#                     posedist = (cam_centers[b].unsqueeze(0) - points).square().sum(-1).sqrt().mean()
                    
#                     d_map = depth_maps[b].clone()
#                     max_val = d_map[valid_mask].max() if valid_mask.any() else 1.0
#                     d_map[~valid_mask] = max_val * 1.2
                    
#                     # Normalize for CV
#                     d_min, d_max = d_map.min(), d_map.max()
#                     d_norm = (d_map - d_min) / (d_max - d_min + 1e-6)
#                     d_img = (d_norm.cpu().numpy() * 255).astype(np.uint8)
                    
#                     edges = cv.Canny(d_img, 50, 150)
#                     contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#                     edge_score = len(contours)
                    
#                     # fixed_cd: completeness, posedist: closeness, edge_score: topology
#                     loss = fixed_cd + (pose_w * posedist) + (edge_w * edge_score)
#                     # print(f"FIXCD {fixed_cd} posedist {posedist} edgescore {edge_score}")
#                 else:
#                     loss = torch.tensor(float('inf'))
                
#                 if loss < best_loss:
#                     best_loss = loss
#                     best_elev = chunk_elevs[b].item()
#                     best_azim = chunk_azims[b].item()
                    
#                     # Prepare debug image (convert to BGR for colorful contours)
#                     # import random
#                     # color_img = cv.cvtColor(d_img, cv.COLOR_GRAY2BGR)
#                     # for cnt in contours:
#                     #     cv.drawContours(color_img, [cnt], -1, (random.randint(0,255), 
#                     #                                           random.randint(0,255), 
#                     #                                           random.randint(0,255)), 1)
#                     # best_img_to_save = color_img
            
#             del pcd_batch, fragments, cameras, rasterizer
#             torch.cuda.empty_cache() 

#         # Hierarchical search update
#         interv = (endv - startv) / num
#         interh = (endh - starth) / num
#         startv, endv = best_elev - interv, best_elev + interv
#         starth, endh = best_azim - interh, best_azim + interh
#         best_final_elev, best_final_azim = best_elev, best_azim
        
#         # if best_img_to_save is not None:
#         #     cv.imwrite(os.path.join(
#         #                 renders_dir, f"combined-{j}-{best_elev:.1f}-{best_azim:.1f}.jpg")
#         #                ,
#         #                best_img_to_save)

#     print(f"Optimal POV Found -> Azimuth: {best_final_azim:.1f}°, Elevation: {best_final_elev:.1f}°")
#     return best_final_elev, best_final_azim

import open3d as o3d
import numpy as np
import torch

def render_with_open3d(pcd, best_elev, best_azim, H=640, W=640):
    """
    pcd_tensor: (N, 3) tensor or numpy array
    best_elev, best_azim: in degrees
    """
    pcd.estimate_normals() # Required for the "clay" / lit look
    
    center = pcd.get_center()
    extent = pcd.get_max_bound() - pcd.get_min_bound()
    distance = np.linalg.norm(extent) * 1.5
    
    # Convert degrees to radians
    theta = np.radians(90 - best_elev) # Elevation from XY plane
    phi = np.radians(best_azim + 180)
    
    # Spherical to Cartesian
    eye = [
        distance * np.sin(theta) * np.cos(phi) + center[0],
        distance * np.sin(theta) * np.sin(phi) + center[1],
        distance * np.cos(theta) + center[2]
    ]
    
    pcd.orient_normals_towards_camera_location(camera_location=eye)
    
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.7, 0.7, 0.7, 1.0]
    material.base_roughness = 0.5 
    material.base_metallic = 0.0
    
    render.scene.add_geometry("pcd", pcd, material)
    render.scene.set_background([0, 0, 0, 1]) # Black background
    
    # vertical_field_of_view, center, eye, up
    render.setup_camera(60.0, center, eye, [0, 1, 0])
    
    image = render.render_to_image()
    depth = render.render_to_depth_image(z_in_view_space=True)
    
    return image, depth

def get_reference_image(pcd, best_elev, best_azim):
    print("Rendering PyTorch3D Reference Image and Depth...")
    
    img, depth = render_with_open3d(pcd, best_elev, best_azim)
    return img
    
    # Unpack both the images and the depth tensor
    # batched_imgs.save(os.path.join(renders_dir, "REFERENCE-pre.png"))
    
    # # ref_rgb = batched_imgs[0].cpu().numpy()
    # ref_rgb = batched_imgs
    # ref_rgb = (ref_rgb * 255).astype(np.uint8)
    # # import cv
    # ref_alpha = torch.zeros_like(depth_tensor[0], dtype=torch.uint8)
    # ref_alpha[depth_tensor[0] > 0] = 255
    # ref_alpha = ref_alpha.cpu().numpy()[..., None] # Add channel dimension
    # ref_rgba = np.concatenate([ref_rgb, ref_alpha], axis=-1)
    
    # best_pov_image = Image.fromarray(ref_rgba, mode='RGBA')
    # best_pov_image.save(os.path.join(renders_dir, "REFERENCE-post.png"))
    
    # return best_pov_image
    
import open3d as o3d

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
    # R = partial_pcd.get_rotation_matrix_from_xyz((np.pi / 2, np.pi/2, 0))
    # partial_pcd.rotate(R, center=(0, 0, 0))
    
    
    
    angles = {
        "stanford-bunny.ply": (12.016324043273926, -129.30612182617188),
        "cow.ply": (10.44897747039795,-2.3510241508483887),
    }
    
    # print("FIND AZIM/ELEV;")
    if object in angles:
        best_elev, best_azim = angles[object]
    else:
        best_elev, best_azim = find_best_reference_pov_full(partial_pcd)
    
    # best_elev = 12.016324043273926
    # best_azim = -129.30612182617188

    print("GET BEST RGB;")
    canonical_image = get_reference_image(partial_pcd, best_elev, best_azim)
    o3d.io.write_image(os.path.join(renders_dir, "RENDER-pre.png"), canonical_image)
    
    