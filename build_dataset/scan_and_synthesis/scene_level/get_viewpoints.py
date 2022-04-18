import numpy as np
import os
import trimesh
import math
import argparse
listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]

#############################################
# Get uniformly sampled points from unit sphere
#############################################
def sample_sphere_uniform(N):
    # N : the number of axis 
    # return: N 3 np.array
    axis = []
    phi  = (5**0.5 - 1) / 2  # 0.618
    for i in range(int(N)):
        z = float(2*i)/N - 1
        x = (1 - z**2)**0.5 * math.cos(2*math.pi*i*phi)
        y = (1 - z**2)**0.5 * math.sin(2*math.pi*i*phi)
        axis.append([x,y,z])
    return np.array(axis)


#############################################
# Judge where points in bbox
#############################################
def not_in_bbox(bounds, vps):
    vps_points = vps[:,:3]
    mask = (bounds[0]<=vps_points)&(vps_points<=bounds[1])
    # print(vps[:10], '\n',bounds, '\n',mask[:10],'\n',~(mask.all(axis=1))[:10])
    vps_not_in = vps_points[~(mask.all(axis=1)), :]  # The points not in the bbox
    if len(vps_not_in) == len(vps_points):
        return True
    else:
        return False 


def main(Scene_Input_Mesh, Out_VPS):
    #-------------------------------------------------
    # Step 1: get the bbox of the room and the scanner
    #-------------------------------------------------
    mesh = trimesh.load(Scene_Input_Mesh)
    mesh_bounds = mesh.bounds
    # print(mesh_bounds)
    scan_bounds = np.ones_like(mesh_bounds)
    scan_bounds[0] = mesh_bounds[0] + 0.01  # keep safe margin
    scan_bounds[1] = mesh_bounds[1] - 0.01
    # print(scan_bounds)

    #-------------------------------------------------
    # Step 2: split the bbox of the scanner
    #-------------------------------------------------
    split_step = 0.5
    split_size = 1.0

    Dx = math.ceil((scan_bounds[1][0] - scan_bounds[0][0])/split_step)  # Keep the last cubes
    Dy = math.ceil((scan_bounds[1][1] - scan_bounds[0][1])/split_step)
    Dz = math.ceil((scan_bounds[1][2] - scan_bounds[0][2])/split_step)

    # Dx = int((scan_bounds[1][0] - scan_bounds[0][0])/split_step)        # Drop the last cubes
    # Dy = int((scan_bounds[1][1] - scan_bounds[0][1])/split_step)
    # Dz = int((scan_bounds[1][2] - scan_bounds[0][2])/split_step)

    # Dx = int(round((scan_bounds[1][0] - scan_bounds[0][0])/split_step)) # Drop the last cubes by round
    # Dy = int(round((scan_bounds[1][1] - scan_bounds[0][1])/split_step))
    # Dz = int(round((scan_bounds[1][2] - scan_bounds[0][2])/split_step))

    # Dx = int((scan_bounds[1][0] - scan_bounds[0][0])/split_step)          # Keep the last cubes only along the z axis by round
    # Dy = int((scan_bounds[1][1] - scan_bounds[0][1])/split_step)
    # Dz = int(round((scan_bounds[1][2] - scan_bounds[0][2])/split_step))

    print("The split step is: {}".format(split_step), '--->' ,"Dx x Dy x Dz is：{} x {} x {}".format(Dx, Dy, Dz))
    scan_bounds_lists = []
    for dx in range(Dx):
        for dy in range(Dy):
            for dz in range(Dz):
                split_bbox = np.ones_like(scan_bounds)
                split_bbox[0][0] = scan_bounds[0][0] + split_step*dx
                split_bbox[0][1] = scan_bounds[0][1] + split_step*dy
                split_bbox[0][2] = scan_bounds[0][2] + split_step*dz
                split_bbox[1][0] = min(split_bbox[0][0] + split_size, scan_bounds[1][0])
                split_bbox[1][1] = min(split_bbox[0][1] + split_size, scan_bounds[1][1])
                split_bbox[1][2] = min(split_bbox[0][2] + split_size, scan_bounds[1][2])
                scan_bounds_lists.append(split_bbox)

    #-------------------------------------------------
    # Step 3: drop the bbox, where something in it
    #-------------------------------------------------
    print(len(scan_bounds_lists))
    scan_bounds_lists_filter = []
    scan_bounds_lists_filter_drop = []
    sample_points, _ = trimesh.sample.sample_surface(mesh, count=50000) 
    for bb in scan_bounds_lists:
        # print(bb)
        judge_in = not_in_bbox(bb, sample_points)
        if np.min(bb[1]-bb[0]) > 0.1:
            if judge_in:
                scan_bounds_lists_filter.append(bb)
            else:
                scan_bounds_lists_filter_drop.append(bb)       
    print(len(scan_bounds_lists_filter), len(scan_bounds_lists_filter_drop))
    
    #-------------------------------------------------
    # Step 4: sample vps in the bbox, also the target points
    #-------------------------------------------------
    vps_in_bbox = []
    sphere_points = 100
    for bb in scan_bounds_lists_filter:
        vpspoints = (bb[0] + bb[1])*0.5
        vps_in_bbox.append(vpspoints)
    vps_sample = np.array(vps_in_bbox).reshape(-1,3)
    vps_with_target = []
    for i in range(len(vps_sample)):
        sphere_vp = sample_sphere_uniform(sphere_points)
        _vps_shpere = np.concatenate((np.repeat([vps_sample[i]], sphere_points, axis=0), sphere_vp), axis=1)
        vps_with_target.append(_vps_shpere)
    vp_sphere_ = np.array(vps_with_target).reshape(-1,6)
    np.savetxt(Out_VPS, vp_sphere_, delimiter=' ', fmt='%f')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='!!!Get the Viewpoints of the synthetic scenes!!!')
    parser.add_argument('--In_Dir', '-id', type=str, default='meshes')
    parser.add_argument('--Out_Dir', '-od', type=str, default='viewpoints')
    args = parser.parse_args()
    print(args)

    In_dir = os.path.abspath(args.In_Dir)
    if args.Out_Dir is not None:
        Out_Dir = os.path.abspath(args.Out_Dir)
    else:
        Out_Dir = In_dir
    os.makedirs(Out_Dir, exist_ok=True)

    print("Processing data dir is: ", In_dir)
    files = listfiles(In_dir)
    for fileone in files:
        print(fileone)
        if os.path.basename(fileone)[-4:] == '.ply':
            main(fileone, fileone.replace('ply','xyz').replace(In_dir, Out_Dir))