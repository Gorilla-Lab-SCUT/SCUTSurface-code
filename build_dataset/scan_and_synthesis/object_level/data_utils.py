import numpy as np
import os
import torch
import trimesh
import open3d as o3d
import subprocess
import multiprocessing
import trimesh.transformations as ts
from scipy.spatial import cKDTree

###################################################
#multiprocess
###################################################
def mp_worker(call):
    """
    Small function that starts a new thread with a system call. Used for thread pooling.
    :param call:
    :return:
    """
    call = call.split(' ')
    verbose = call[-1] == '--verbose'
    if verbose:
        call = call[:-1]
        subprocess.run(call, shell=False)
    else:
        #subprocess.run(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # suppress outputs
        subprocess.run(call, stdout=subprocess.DEVNULL, shell=False)

def start_process_pool(worker_function, parameters, num_processes, timeout=None):
    if len(parameters) > 0:
        if num_processes <= 1:
            print('Running loop for {} with {} calls on {} workers'.format(
                str(worker_function), len(parameters), num_processes))
            results = []
            for c in parameters:
                results.append(worker_function(*c))
            return results
        print('Running loop for {} with {} calls on {} subprocess workers'.format(
            str(worker_function), len(parameters), num_processes))
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.starmap(worker_function, parameters)
            return results
    else:
        return None


###################################################
#read path name 
###################################################
def path_name(file_dir):
    for _, paths, _ in os.walk(file_dir):
        return(paths)
###################################################
#read file name 
###################################################
def file_name(file_dir):
    for _,_, files in os.walk(file_dir):
        return(files)


###########################################
# Farthest Point Sample
###########################################
def farthest_point_sample_torch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, _ = xyz.shape
    centroids = torch.zeros((npoint,), dtype=torch.long).cuda()
    distance = torch.ones((N,), dtype=torch.float32).cuda() * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).cuda()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, axis=-1)[1]
    return centroids

def farthest_point_sample(point, npoint):
    try:
        # np.random.shuffle(point)
        if len(point) > 5000000:
            print('Too much points, conduct random sample first')
            point = randomsample(point, 5000000)
        xyz = torch.from_numpy(point).type(torch.float32).cuda('cuda:0')
        idx = farthest_point_sample_torch(xyz[:,:3], npoint)
        return xyz[idx].cpu().numpy()
    except:
        print('Warning!!! Could not conduct FPS, replaced with Random')
        if len(point) > npoint:
            ptsidx = np.random.choice(len(point), npoint, replace=False)
        else:
            print('Warning!!! Not enough Points')
            ptsidx = np.random.choice(len(point), npoint, replace=True)
        return point[ptsidx]


###########################################
# Random Sample
########################################### 
def randomsample(points, npoints=100000):
    try:
        choice = np.random.choice(len(points), npoints, replace=False)
    except:
        choice = np.random.choice(len(points), npoints, replace=True)
    points = points[choice,:]
    return points


###########################################
# how many points
########################################### 
def howmuchpoint(filename, missing):
    fflen = 0
    if 'complex' in filename:
        fflen = 160000 * (1-missing)
    elif 'ordinary' in filename:
        fflen = 120000 * (1-missing)
    else:
        fflen = 80000 * (1-missing)
    return fflen


###########################################
# Outlier disturb
########################################### 
def outlierbpoints(points, number, intensity = 0.15):
    number = len(points) * number
    idx = np.random.choice(len(points), int(number), replace=False)
    points = points[idx]
    points[:,0] += np.random.uniform(0.01, intensity, len(points))*(2*np.random.randint(0,2,size=len(points))-1)
    points[:,1] += np.random.uniform(0.01, intensity, len(points))*(2*np.random.randint(0,2,size=len(points))-1)
    points[:,2] += np.random.uniform(0.01, intensity, len(points))*(2*np.random.randint(0,2,size=len(points))-1)
    return points


###########################################
# disturb points
########################################### 
def disturbpoints(points, angle, intensity):
    angle = angle * (np.pi /180)
    alpha = np.random.uniform(-angle, angle)
    beta = np.random.uniform(-angle, angle)
    gamma = np.random.uniform(-angle, angle)
    Re = trimesh.transformations.euler_matrix(alpha, beta, gamma, 'rxyz')
    if points.shape[1] == 3:
        points = np.array(ts.transform_points(points[:,:3], Re))
    else:
        points = np.concatenate((np.array(ts.transform_points(points[:,:3], Re)), np.array(ts.transform_points(points[:,3:], Re))),axis=-1)
    points[:,0] += np.random.uniform(-intensity, intensity)
    points[:,1] += np.random.uniform(-intensity, intensity)
    points[:,2] += np.random.uniform(-intensity, intensity)
    return points


###########################################
# discard viewpoints by tracket
########################################### 
def xyztoSpherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(xyz[:,1], xyz[:,0]) + np.pi
    ptsnew[:,5] = np.arctan2(np.sqrt(xy), xyz[:,2])     # for elevation angle defined from Z-axis down
    ptsnew[:,4:] = ptsnew[:,4:] / np.pi * 180
    return ptsnew[:,4:]

def discardvpsbytracket(viewpoints, zanglelist):
    zangle = xyztoSpherical(viewpoints)[:,-1]
    return [i for i, z in enumerate(zangle) 
        if any([mn <= z <= mx for mn, mx in zanglelist])]


###########################################
# get nearest neighbour
###########################################
def kdtreeneighbor(sourcepoint, points, radius=0.5):
    points = np.insert(np.array(points), 0, values=np.array(sourcepoint), axis=0)
    tree = cKDTree(points, compact_nodes=False, balanced_tree=False)
    npp = tree.query_ball_point(points[0],radius)
    return npp


########################################################################
# Reject sample too far 
########################################################################
def judge_far_away_by_threshold(pts, target, dist_thres=2.0, percent_thres=0.6):
    total = len(pts)
    in_dist = len(kdtreeneighbor(target[:3], pts[:,:3], dist_thres))
    if in_dist/total >= percent_thres:
        return True
    else:
        return False