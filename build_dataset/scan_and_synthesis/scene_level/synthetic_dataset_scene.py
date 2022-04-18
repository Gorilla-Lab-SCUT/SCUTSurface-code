import data_utils
import numpy as np
import os
from tqdm import tqdm
import open3d as o3d
import argparse

listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]

########################################################################
# Normal Estimate
########################################################################
def extimate_normals_and_orientation_with_o3d(pc,knn=40,orientation_reference=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn), fast_normal_computation=True)
    pcd.orient_normals_towards_camera_location(orientation_reference[:3])
    return np.concatenate((pcd.points, pcd.normals), axis=-1)

########################################################################
# To estimate normal batch and generate the outliers
########################################################################
def estimate_one(ptsssss_file, viewpoint_file, concate_file, outliers_file=None, string=None, anoise=0):
    ptsssss_file = ptsssss_file.replace('\\','/').replace('//','/')
    print(string)
    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = np.arange(len(viewpoint))
    for idx in tqdm(Index):
        if anoise==0:
            ptsssss_file_one = os.path.join(ptsssss_file, ptsssss_file.split('/')[-1] +"_"+str(idx+1)+'.txt') 
        else:
            ptsssss_file_one = os.path.join(ptsssss_file, ptsssss_file.split('/')[-1] +"_"+str(anoise)+"_"+str(idx+1)+'.txt') 
        pts = np.loadtxt(ptsssss_file_one, delimiter=' ')[:,:3]      
        vps = viewpoint[idx]
        ptsn = extimate_normals_and_orientation_with_o3d(pts, knn=40, orientation_reference=vps)
        outfile = ptsssss_file_one.replace(ptsssss_file, concate_file)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.savetxt(outfile, ptsn , fmt="%.6f",delimiter=' ')
        with open(outliers_file.replace('.txt','_p0004i01.txt'), 'ab') as f:
            outliers = data_utils.outlierbpoints(ptsn, 0.02, intensity=0.1)   # 2.5%   0.1m
            np.savetxt(f, outliers, delimiter=' ', fmt='%.6f')
        f.close()

def Get_estimate_noise(anoise, In_Dir, In_non_dir, Out_Dir, num_work=1):
    cmds = []
    fileslen = len(data_utils.path_name(In_Dir))
    for i in range(fileslen):
        pathobj = data_utils.path_name(In_Dir)[i]
        ptsssss_file = os.path.join(In_Dir, str(pathobj))
        # print(ptsssss_file)
        viewpoint_file = os.path.join(In_non_dir, str(pathobj) ,"viewpoint.txt")
        concate_file = ptsssss_file.replace(os.path.abspath(In_Dir), os.path.abspath(Out_Dir))
        outliers_file = os.path.join(concate_file, 'outliers_'+str(anoise)+'.txt')
        os.makedirs(os.path.dirname(concate_file),exist_ok=True)
        string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
        cmds.append((ptsssss_file, viewpoint_file, concate_file,  outliers_file, string, anoise))
    data_utils.start_process_pool(estimate_one, cmds, num_work)

########################################################################
# To estimate normal batch
########################################################################
def Merge_one(ptsssss_file, pathobj, concate_file, anoise, Misalignment_angle, Misalignment_intensity, total_points,outlier_portion, string=None):
    if string is not None:
        print(string)
    pts_list = []
    out_list = []
    print(pathobj)
    for pts_one in tqdm(listfiles(ptsssss_file)):
        if 'outliers_'+str(anoise)+'_p0004i01' in pts_one:
            pts = np.loadtxt(pts_one)
            out_list.append(data_utils.randomsample(pts, int(total_points*outlier_portion)))
        elif pathobj +"_"+str(anoise) in pts_one:
            print(pts_one)
            pts = np.loadtxt(pts_one)
            print(len(pts))
            pts_list.append(data_utils.disturbpoints(pts, Misalignment_angle, Misalignment_intensity))
        else:
            pass
    pts_list = np.concatenate(np.array(pts_list), axis=0)
    out_list = np.asarray(out_list).squeeze()
    ptsc_len = int(total_points*(1-outlier_portion))
    ptsc = np.concatenate((data_utils.randomsample(pts_list, ptsc_len), out_list), axis=0)
    np.savetxt(concate_file, ptsc, fmt='%.6f', delimiter=' ')


def Meger_all_frame(In_Dir, Out_Dir, anoise=0.005, total_points=1000000, outlier_portion=0.004, Misalignment_intensity=0.02, Misalignment_angle=2, num_work=1):
    cmds = []
    fileslen = len(data_utils.path_name(In_Dir))
    for i in range(fileslen):
        pathobj = data_utils.path_name(In_Dir)[i]
        ptsssss_file = os.path.join(In_Dir, str(pathobj))
        concate_file = os.path.join(Out_Dir, pathobj+'_'+str(anoise)+'_pcs.xyz')
        os.makedirs(os.path.dirname(concate_file), exist_ok=True)
        string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
        cmds.append((ptsssss_file, pathobj, concate_file, anoise,Misalignment_angle, Misalignment_intensity, total_points, outlier_portion, string))
    data_utils.start_process_pool(Merge_one, cmds, num_work)


##################################################################################
# Do the generation
##################################################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='!!!Final generation!!!')
    parser.add_argument('--In_Noise_Dir', '-inoid', type=str, default='Noise')
    parser.add_argument('--In_Nonuniform_Dir', '-inond', type=str, default='Nonuniform')
    parser.add_argument('--Out_Noise_Normal_Dir', '-onoind', type=str, default='Noise_Normal')
    parser.add_argument('--Out_Dir', '-od', type=str, default='Out')
    parser.add_argument('--num_worker', '-nw', type=int, default=1)

    args = parser.parse_args()
    print(args)

    In_Noise_Dir = os.path.abspath(args.In_Noise_Dir)
    In_Nonuniform_Dir = os.path.abspath(args.In_Nonuniform_Dir)
    Out_Noise_Normal_Dir = os.path.abspath(args.Out_Noise_Normal_Dir)

    Get_estimate_noise(0.005, In_Noise_Dir,In_Nonuniform_Dir,Out_Noise_Normal_Dir, num_work=args.num_worker)

    Out_Dir = os.path.abspath(args.Out_Dir)

    Meger_all_frame(Out_Noise_Normal_Dir, Out_Dir, anoise=0.005, total_points=1000000, outlier_portion=0.004, Misalignment_intensity=0.015, Misalignment_angle=1.5, num_work=args.num_worker)