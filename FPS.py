import os
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

def farthest_point_sampling(xyz, npoint):
    N, _ = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long)
    distance = torch.ones(N) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long)[0]
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest].unsqueeze(0)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance)
    return centroids

def process_npy_file(input_path, output_path, npoint=1024):
    data = np.load(input_path)  # shape (8192, 6) or (8192, 3)
    xyz = torch.tensor(data[:, :3], dtype=torch.float32)
    sampled_idx = farthest_point_sampling(xyz, npoint)
    sampled_xyz = xyz[sampled_idx].numpy()

    if data.shape[1] == 6:  # has RGB
        rgb = data[:, 3:]
        sampled_rgb = rgb[sampled_idx.numpy()]
        output = np.concatenate([sampled_xyz, sampled_rgb], axis=1)
    else:
        output = sampled_xyz

    np.save(output_path, output)

def process_ply_file(input_path, output_path, npoint=2048):
    pcd = o3d.io.read_point_cloud(input_path)
    xyz = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
    sampled_idx = farthest_point_sampling(xyz, npoint)
    sampled_xyz = xyz[sampled_idx].numpy()

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_xyz)

    # 保留颜色（如果有）
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        sampled_colors = colors[sampled_idx.numpy()]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

    o3d.io.write_point_cloud(output_path, downsampled_pcd)

def batch_fps_downsample(input_dir, output_dir, npoint=2048):
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)

    for file in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        if file.endswith(".npy"):
            process_npy_file(input_path, output_path, npoint)
        elif file.endswith(".ply"):
            process_ply_file(input_path, output_path, npoint)

    print("Finished ！！！")


input_folder = " "
output_folder = " "

batch_fps_downsample(input_folder, output_folder)
