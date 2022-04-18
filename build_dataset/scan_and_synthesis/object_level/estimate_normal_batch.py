import numpy as np
import os
from tqdm import tqdm
import data_utils
import open3d as o3d
import shutil
import argparse

########################################################################
# Normal Estimate
########################################################################
def extimate_normals_and_orientation_with_o3d(pc,knn=40,orientation_reference=None):#, radius=0.1, max_nn=30, orientation_reference):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn), fast_normal_computation=True)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # pcd.orient_normals_to_align_with_direction(orientation_reference=orientation_reference)
    pcd.orient_normals_towards_camera_location(camera_location=orientation_reference) 
    return np.concatenate((pcd.points, pcd.normals), axis=-1)

########################################################################
# To estimate normal batch
########################################################################
def estimate_one(fileidx_file, ptsssss_file, viewpoint_file, concate_file, string=None):
    print(string)
    fileidx = np.loadtxt(fileidx_file, delimiter=' ')
    ptsssss = np.loadtxt(ptsssss_file, delimiter=' ')
    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = np.arange(len(viewpoint))
    with open(concate_file, 'ab') as f:
        for idx in Index:
            vps = viewpoint[idx]
            pts = ptsssss[int(fileidx[idx][1]):int(fileidx[idx][2])]
            ptsn = extimate_normals_and_orientation_with_o3d(pts, knn=40, orientation_reference=vps)
            np.savetxt(f, ptsn , fmt="%f",delimiter=' ')
    f.close()

def Get_estimate_nonuniform(In_Dir, Out_Dir, num_work=1):
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    cmds = []
    i = 0
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+".txt")
            viewpoint_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,"viewpoint.txt")
            concate_file = ptsssss_file.replace(os.path.abspath(In_Dir), os.path.abspath(Out_Dir))
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            shutil.copyfile(fileidx_file, fileidx_file.replace(os.path.abspath(In_Dir), os.path.abspath(Out_Dir)))
            shutil.copyfile(viewpoint_file, viewpoint_file.replace(os.path.abspath(In_Dir), os.path.abspath(Out_Dir)))
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file,  string))
            i += 1
    data_utils.start_process_pool(estimate_one, cmds, num_work)


def Get_estimate_noise(anoise, In_Dir, In_non_dir, Out_Dir, num_work=1):
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    cmds = []
    i = 0
    print(In_Dir)
    print(data_utils.path_name(In_Dir))
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        print(pathsmc)
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_"+str(anoise)+".txt")
            viewpoint_file = os.path.join(In_non_dir, pathsmc, str(pathobj) ,"viewpoint.txt")
            concate_file = ptsssss_file.replace(os.path.abspath(In_Dir), os.path.abspath(Out_Dir))
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            shutil.copyfile(fileidx_file, fileidx_file.replace(os.path.abspath(In_Dir), os.path.abspath(Out_Dir)))
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file,  string))
            i += 1
    data_utils.start_process_pool(estimate_one, cmds, num_work)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='!!!Final generation!!!')
    parser.add_argument('--In_Nonuniform_Dir', '-inon', type=str, default='../../../data/synthetic_object/Nonuniform')
    parser.add_argument('--Out_Nonuniform_Normal_Dir', '-onon', type=str, default='../../../data/synthetic_object/Nonuniform_Normal')
    parser.add_argument('--In_Noise_Dir', '-inoi', type=str, default='../../../data/synthetic_object/Noise')
    parser.add_argument('--Out_Noise_Normal_Dir', '-onoi', type=str, default='../../../data/synthetic_object/Noise_Normal')
    parser.add_argument('--num_worker', '-nw', type=int, default=1)

    args = parser.parse_args()
    print(args)

    In_Nonuniform_Dir = os.path.abspath(args.In_Nonuniform_Dir)
    Out_Nonuniform_Normal_Dir = os.path.abspath(args.Out_Nonuniform_Normal_Dir)

    In_Noise_Dir = os.path.abspath(args.In_Noise_Dir)
    Out_Noise_Normal_Dir = os.path.abspath(args.Out_Noise_Normal_Dir)

    Get_estimate_nonuniform(In_Nonuniform_Dir, Out_Nonuniform_Normal_Dir, num_work=args.num_worker)
    # 0.001  0.003  0.006  
    Get_estimate_noise(0.001, In_Noise_Dir, In_Nonuniform_Dir, Out_Noise_Normal_Dir, num_work=args.num_worker)
    Get_estimate_noise(0.003, In_Noise_Dir, In_Nonuniform_Dir, Out_Noise_Normal_Dir, num_work=args.num_worker)
    Get_estimate_noise(0.006, In_Noise_Dir, In_Nonuniform_Dir, Out_Noise_Normal_Dir, num_work=args.num_worker)
