import shutil
import time
from tensorboardX import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
import trimesh
import math
import pandas as pd
import torch
import random
from scipy.spatial import cKDTree as KDTree

##############################################################
# Utils
##############################################################
def load_xyz(file_path):
    data = np.loadtxt(file_path, delimiter=' ').astype('float32')
    return data

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0
    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n
    def item(self):
        return self.v

_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

class Timer():
    def __init__(self):
        self.v = time.time()
    def s(self):
        self.v = time.time()
    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def sampleGT(filename, samplepointsnum):
    mesh = trimesh.load(filename)
    mesh.fix_normals()  # ensuring outward normals
    sample_random, index_random = trimesh.sample.sample_surface(mesh, samplepointsnum)
    sample_normal = mesh.face_normals[index_random]
    return sample_random, sample_normal

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def writecsv(filename, data):
    pf = pd.DataFrame([data])
    pf = pf.fillna(-1)
    with open(filename, 'a+', newline='', encoding='utf-8') as f:
        try:
            pd.read_csv(filename, skiprows=1)
            flag =  0
        except:
            flag = 1
        if flag:
            pf.to_csv(f, mode='a+')
        else:
            pf.to_csv(f, mode='a+',header=False)

##############################################################
# Build Patch Dataset
##############################################################
def voxelize(point_cloud, leaf_size, point_per_patch, out_file=None):
    x_min, y_min, z_min = np.amin(point_cloud[:,:3], axis=0) #计算x y z 三个维度
    x_max, y_max, z_max = np.amax(point_cloud[:,:3], axis=0)
 
    Dx = (x_max - x_min)//leaf_size + 1
    Dy = (y_max - y_min)//leaf_size + 1
    Dz = (z_max - z_min)//leaf_size + 1
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))
 
    h = [(pc[0]-x_min)//leaf_size + ((pc[1]-y_min)//leaf_size)*Dx + ((pc[2]-z_min)//leaf_size)*Dx*Dy for pc in point_cloud]

    voxel_dix = np.unique(h)
    patch_list = []
    for idx in voxel_dix:
        idx_patch = point_cloud[h==idx]
        if len(idx_patch) < 100:
            pass
        else:
            # print(len(idx_patch))
            choice = np.random.choice(len(idx_patch), int(point_per_patch), replace=True)
            if out_file is None:
                patch_list.append(idx_patch[choice])
            else:
                np.savetxt(out_file +'_'+str(int(idx)) +'.xyz',idx_patch[choice], delimiter=' ', fmt='%.6f')
    if len(patch_list) > 0:
        return patch_list


def voxelize_for_test(pc_pred, pc_gt, leaf_size):
    x_min_p, y_min_p, z_min_p = np.amin(pc_pred[:,:3], axis=0)
    x_max_p, y_max_p, z_max_p = np.amax(pc_pred[:,:3], axis=0)
    x_min_g, y_min_g, z_min_g = np.amin(pc_gt[:,:3], axis=0)
    x_max_g, y_max_g, z_max_g = np.amax(pc_gt[:,:3], axis=0)

    x_min, y_min, z_min= np.amin([[x_min_p, y_min_p, z_min_p],[x_min_g, y_min_g, z_min_g]], axis=0)
    x_max, y_max, z_max= np.amax([[x_max_p, y_max_p, z_max_p],[x_max_g, y_max_g, z_max_g]], axis=0)

    x_min = math.floor(x_min/leaf_size) * leaf_size
    y_min = math.floor(y_min/leaf_size) * leaf_size
    z_min = math.floor(z_min/leaf_size) * leaf_size
    x_max = math.ceil(x_max/leaf_size) * leaf_size
    y_max = math.ceil(y_max/leaf_size) * leaf_size
    z_max = math.ceil(z_max/leaf_size) * leaf_size

    Dx = (x_max - x_min)//leaf_size
    Dy = (y_max - y_min)//leaf_size
    Dz = (z_max - z_min)//leaf_size
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

    h_pred = np.array([(pc[0]-x_min)//leaf_size + ((pc[1]-y_min)//leaf_size)*Dx + ((pc[2]-z_min)//leaf_size)*Dx*Dy for pc in pc_pred])
    h_gt = np.array([(pc[0]-x_min)//leaf_size + ((pc[1]-y_min)//leaf_size)*Dx + ((pc[2]-z_min)//leaf_size)*Dx*Dy for pc in pc_gt])

    patch_list_pair = []
    for i in tqdm(range(int(Dx*Dy*Dz)), leave=False, desc='build paris'):
        patch_pred = pc_pred[np.where(h_pred==i)]
        patch_gt = pc_gt[np.where(h_gt==i)]
        if len(patch_pred) == 0 and len(patch_gt) == 0:
            pass
        else:
            outdict = {
                'pred_pts' : patch_pred,
                'gt_pts' : patch_gt,
            } 
            patch_list_pair.append(outdict)
    return patch_list_pair  

def voxelize_for_test_overlap(pc_pred, pc_gt, leaf_size):
    x_min_p, y_min_p, z_min_p = np.amin(pc_pred[:,:3], axis=0)
    x_max_p, y_max_p, z_max_p = np.amax(pc_pred[:,:3], axis=0)
    x_min_g, y_min_g, z_min_g = np.amin(pc_gt[:,:3], axis=0)
    x_max_g, y_max_g, z_max_g = np.amax(pc_gt[:,:3], axis=0)

    x_min, y_min, z_min= np.amin([[x_min_p, y_min_p, z_min_p],[x_min_g, y_min_g, z_min_g]], axis=0)
    x_max, y_max, z_max= np.amax([[x_max_p, y_max_p, z_max_p],[x_max_g, y_max_g, z_max_g]], axis=0)

    x_min = math.floor(x_min/leaf_size) * leaf_size
    y_min = math.floor(y_min/leaf_size) * leaf_size
    z_min = math.floor(z_min/leaf_size) * leaf_size
    x_max = math.ceil(x_max/leaf_size) * leaf_size
    y_max = math.ceil(y_max/leaf_size) * leaf_size
    z_max = math.ceil(z_max/leaf_size) * leaf_size

    Dx = (x_max - x_min)//leaf_size
    Dy = (y_max - y_min)//leaf_size
    Dz = (z_max - z_min)//leaf_size
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

    h_pred = np.array([(pc[0]-x_min)//leaf_size + ((pc[1]-y_min)//leaf_size)*Dx + ((pc[2]-z_min)//leaf_size)*Dx*Dy for pc in pc_pred])
    h_gt = np.array([(pc[0]-x_min)//leaf_size + ((pc[1]-y_min)//leaf_size)*Dx + ((pc[2]-z_min)//leaf_size)*Dx*Dy for pc in pc_gt])

    patch_list_pair = []

    for i in tqdm(range(int(Dx*Dy*Dz)), leave=False, desc='build paris'):
        patch_pred = pc_pred[np.where(h_pred==i)]
        patch_gt = pc_gt[np.where(h_gt==i)]
        if len(patch_pred) == 0 and len(patch_gt) == 0:
            pass
        else:
            outdict = {
                'pred_pts' : patch_pred,
                'gt_pts' : patch_gt,
            } 
            patch_list_pair.append(outdict)
    # Overlap
    x_max += leaf_size*0.5
    y_max += leaf_size*0.5
    z_max += leaf_size*0.5
    x_max = math.ceil(x_max/leaf_size) * leaf_size
    y_max = math.ceil(y_max/leaf_size) * leaf_size
    z_max = math.ceil(z_max/leaf_size) * leaf_size
    Dx = (x_max - x_min)//leaf_size
    Dy = (y_max - y_min)//leaf_size
    Dz = (z_max - z_min)//leaf_size
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

    pc_pred += leaf_size*0.5
    pc_gt += leaf_size*0.5

    h_pred_o = np.array([(pc[0]-x_min)//leaf_size + ((pc[1]-y_min)//leaf_size)*Dx + ((pc[2]-z_min)//leaf_size)*Dx*Dy for pc in pc_pred])
    h_gt_o = np.array([(pc[0]-x_min)//leaf_size + ((pc[1]-y_min)//leaf_size)*Dx + ((pc[2]-z_min)//leaf_size)*Dx*Dy for pc in pc_gt])

    for i in tqdm(range(int(Dx*Dy*Dz)), leave=False, desc='build overlap pairs'):
        patch_pred = pc_pred[np.where(h_pred_o==i)]
        patch_gt = pc_gt[np.where(h_gt_o==i)]
        if len(patch_pred) == 0 and len(patch_gt) == 0:
            pass
        else:
            # np.savetxt('test_o/'+str(int(i))+'.xyz', np.concatenate((patch_pred, patch_gt), axis=0), fmt='%.f', delimiter=' ')
            outdict = {
                'pred_pts' : patch_pred,
                'gt_pts' : patch_gt,
            } 
            patch_list_pair.append(outdict)
    return patch_list_pair


def build_patch(shape_root, shape_patch_root, shape_name, voxel_size=10.0, point_per_patch=5000, unit_scale=False):
    print('voxel_size is : ', voxel_size, '\t', 'point per patch is : ', point_per_patch)
    for shape in tqdm(shape_name):
        pts = load_xyz(os.path.join(shape_root, shape +'.xyz'))
        # print(pts)
        if unit_scale:
            scale = np.abs(pts[:,:3]).max()
            pts[:,:3] = pts[:,:3] / scale
        print(np.amax(pts, axis=0), '\t' ,np.amin(pts, axis=0))
        out_file = os.path.join(shape_patch_root, shape)
        voxelize(pts, voxel_size, point_per_patch, out_file)



#################################################################
# Random Non-uniform
#################################################################
# Farthest Point Sample
def farthest_point_sample_torch(points, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    xyz = points[:,:3]
    xyz = torch.from_numpy(xyz).type(torch.float32).cuda('cuda:0')
    N, _ = xyz.shape
    centroids = torch.zeros((npoint,), dtype=torch.long).cuda('cuda:0')
    distance = torch.ones((N,), dtype=torch.float32).cuda('cuda:0') * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).cuda('cuda:0')
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, axis=-1)[1]
    return centroids.cpu().numpy()    #xyz[idx].cpu().numpy()
    # return points[centroids.cpu().numpy()]

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, _ = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # point = point[centroids.astype(np.int32)]
    return centroids.astype(np.int32)

# Random Sample
def randomsample(points, npoints=100000):
    choice = np.random.choice(len(points), npoints, replace=False)
    points = points[choice,:]
    return points

# Find k nearest
def find_k_nearest(points, centroids_idx, k_max):
    points = points[:,:3]
    kdtreee = KDTree(points)
    _, idx = kdtreee.query(points[centroids_idx], k=k_max, n_jobs=8)
    drop_list = []
    for id in idx:
        drop_list.extend(id[0: random.randint(20, k_max-1)])
    drop_list = set(drop_list)
    return drop_list

# Build Nonuniform
def buildnonuniform(points):
    lenpoints = len(points)
    # fps_idx = farthest_point_sample_torch(points, int(lenpoints/100))
    fps_idx = farthest_point_sample(points, int(lenpoints/100))
    drop_fps_len = random.randint(10, len(fps_idx)-1)
    drop_idx = fps_idx[np.random.choice(len(fps_idx), drop_fps_len, replace=False)]
    drop_list = find_k_nearest(points, drop_idx, k_max=150)
    # print(drop_list)
    keep_list = list(set(range(0, lenpoints)) - set(drop_list))
    # print(keep_list)
    if len(keep_list) > 20:
        keep_points =  points[keep_list]
    else:
        keep_points = points[:20, :]
    keep_choice = np.random.choice(len(keep_points), lenpoints, replace=True)
    return keep_points[keep_choice]