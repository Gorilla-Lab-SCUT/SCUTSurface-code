import torch
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from utils import load_config, start_process_pool
from tqdm import tqdm
import time
import argparse

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
    distance = torch.ones((N,), dtype=torch.float32).cuda() * 1e5
    farthest = torch.randint(0, N, (1,), dtype=torch.long).cuda()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, axis=-1)[1]
    del xyz
    del distance
    del farthest
    return centroids


def farthest_point_sample(infile, outfile, npoint, string):
    print(string)
    point = np.loadtxt(infile, delimiter=' ')
    point = point[~np.isnan(point).any(axis=1),:]   #remove nan
    npoint = int(len(point)*npoint)
    print(infile)
    strtime = time.time()
    np.random.shuffle(point)
    xyz = torch.from_numpy(point).type(torch.float32).cuda()
    idx = farthest_point_sample_torch(xyz[:,:3], npoint)
    print('fts time ---->: ', outfile , '----->  ', time.time() - strtime)
    np.savetxt(outfile, xyz[idx].cpu().numpy(), fmt='%f', delimiter=' ')


def main(cfg):
    in_dir = cfg['fps']['in_dir']
    out_dir = cfg['fps']['out_dir']

    os.makedirs(out_dir,exist_ok=True)

    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    files = listfiles(in_dir)

    cmds = []
    for i in tqdm(range(len(files))):
        if files[i].split('.')[-1] == 'txt':
            outfile = files[i].replace(in_dir, out_dir).replace('txt','txt')
            if os.path.exists(outfile):
                pass
            else:
                os.makedirs(os.path.dirname(outfile),exist_ok=True)
                string ='Conducting FPS ----> ' + str(i) +'/'+ str(len(files)) + ' -----> ' + outfile
                cmds.append((files[i], outfile, cfg['fps']['th'], string))
    print("##############################################################################################")
    start_process_pool(farthest_point_sample, cmds, num_processes=cfg['fps']['num_worker'])
    print("FPS finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='!!!Re-sampling!!!')
    parser.add_argument('--configure', '-conf', type=str, default='config_example.yaml', help='the config yaml file')
    args = parser.parse_args()
    cfg = load_config(str(args.configure))
    main(cfg)