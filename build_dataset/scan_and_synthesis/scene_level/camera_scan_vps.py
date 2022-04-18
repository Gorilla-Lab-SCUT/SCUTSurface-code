import bpy
import math
import os
import sys
from numpy import arange
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *
import blensor
import random
import numpy as np
from mathutils import Matrix
from mathutils import Vector

#######################################################################################################
def Camera_Rotate_Scan(file_loc = None,       # the ply or other 3d file input dir
                        store_file = None,    # output dir
                        store_file_view = None,
                        store_file_N = None,
                        #sample times
                        camera_tp = 'tof',                          # Camera types, inference the top info 
                        camera_vps = None,
                        # object scale
                        object_scale = Vector([1.0,1.0,1.0]),       # resize the object
                        # resolution
                        image_x = 50,              
                        image_y = 50,
                        cam_max_dist = 3.5,
                        # Noise parameter
                        Noise_type = 'gaussian',     # ['gaussian','laplace']
                        Noise_mu = 0,                # the average value of the noise
                        Noise_sg = 0,                # the variance valude of the noise 
                        DB_mu = 1,                  
                        DB_sg = 1,
                        Global_Noise_Scale = 0.25,
                        Global_Noise_Smooth = 1.5,
                        # whether store camera normal
                        camera_normal = False):
    
    print(file_loc)
    print(store_file)
    print(store_file_view)
    ###################################################################################################
    imported_object = bpy.ops.import_mesh.ply(filepath=file_loc)
    obj_object = bpy.context.selected_objects[0]
    obj_object.name='Object'
    bpy.data.objects["Object"].scale= object_scale
    obj_object.rotation_mode = 'QUATERNION'

    """set the parameters of the camera, camera type\camera noise """
    camera = bpy.data.objects['Camera'] 
    camera.add_noise_scan_mesh = True
    camera.scan_type = camera_tp
    if camera.scan_type==camera_type[0]:
        camera.velodyne_noise_mu = Noise_mu
        camera.velodyne_noise_sigma = 0
        camera.velodyne_db_noise_mu = DB_mu
        camera.velodyne_db_noise_sigma = DB_sg
        camera.velodyne_noise_type = Noise_type
        
    elif camera.scan_type==camera_type[1]:
        camera.ibeo_noise_mu = Noise_mu
        camera.ibeo_noise_sigma = 0

    elif camera.scan_type==camera_type[2]:
        camera.tof_noise_mu = Noise_mu
        camera.tof_noise_sigma = 0
        camera.tof_lens_angle_h = 70
        camera.tof_lens_angle_w = 60
        camera.tof_xres = image_x
        camera.tof_yres = image_y
        camera.tof_max_dist = cam_max_dist
        camera.tof_focal_length = 3.657   #default 10

    elif camera.scan_type==camera_type[3]:
        camera.kinect_noise_mu = Noise_mu
        camera.kinect_noise_sigma = 0
        camera.kinect_noise_scale = Global_Noise_Scale
        camera.kinect_noise_smooth = Global_Noise_Smooth
        camera.kinect_max_dist = cam_max_dist
        camera.kinect_min_dist = 0.75
        camera.kinect_xres = image_x
        camera.kinect_yres  = image_y
        camera.kinect_flength = 4.73

    elif camera.scan_type==camera_type[4]:
        camera.generic_noise_mu = Noise_mu
        camera.generic_noise_sigma = 0

    elif camera.scan_type==camera_type[5]:
        pass
    ##################################################
    camera.local_coordinates = False     # use world coordinate
    origin = bpy.data.objects['Empty']   # set an origin for the camera to track

    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    """clear all scanning datas  """
    for item in bpy.data.objects:
        if item.type == 'MESH' and item.name.startswith('Scan'):
            bpy.data.objects.remove(item)
    for item in bpy.data.objects:
        if item.type == 'MESH' and item.name.startswith('NoisyScan'):
            bpy.data.objects.remove(item)
    """clear the scanning in view windows and start newly scan"""
    bpy.ops.blensor.delete_scans()

    step = 0
    ptt = 0
    pttn = np.zeros(len(Noise_sg))

    for vps in camera_vps:
        # clear all scanning datas
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('Scan'):
                bpy.data.objects.remove(item)
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                bpy.data.objects.remove(item)
        
        #set camera location, and the target of the camera to tract
        cx = vps[0]; cy = vps[1]; cz = vps[2]
        camera.location[0] = cx
        camera.location[1] = cy
        camera.location[2] = cz
        print('Cam_pose' ,cx, cy, cz)

        ox = vps[3]; oy = vps[4]; oz = vps[5]
        origin.location=Vector([ox,oy,oz])
        print('Target_point', ox, oy, oz)

        camera.data.sensor_height = camera.data.sensor_width
        camera_constraint                     = camera.constraints.new(type="TRACK_TO")
        camera_constraint.track_axis          = "TRACK_NEGATIVE_Z"
        camera_constraint.up_axis             = "UP_Y"
        camera_constraint.target              = origin
        camera.rotation_mode                  = "QUATERNION"
        bpy.context.scene.render.engine       = "CYCLES"
        bpy.context.scene.camera              = bpy.data.objects["Camera"]

        # activate the camera
        bpy.context.scene.objects.active = camera
        for i in range(len(Noise_sg)):
            #clear all scanning datas
            for item in bpy.data.objects:
                if item.type == 'MESH' and item.name.startswith('Scan'):
                    bpy.data.objects.remove(item)
            for item in bpy.data.objects:
                if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                    bpy.data.objects.remove(item)
            camera.tof_noise_sigma = Noise_sg[i]
            # delete the old scan, and create a new scan
            bpy.ops.blensor.delete_scans()
            if i == 0: 
                camera.add_scan_mesh = True
                bpy.ops.blensor.scan()
                # store the data of the current scan
                pointsscan = []
                pointsscannoisy = []
                for item in bpy.data.objects:
                    if item.type == 'MESH' and item.name.startswith('Scan'):
                        for sp in item.data.vertices:
                            if camera_normal == True:
                                pointsscan.append([sp.co, sp.normal])
                            else:
                                pointsscan.append([sp.co])
                        if camera_normal == True:
                            pointsscan = np.reshape(np.array(pointsscan),(-1,6))
                        else:
                            pointsscan = np.reshape(np.array(pointsscan),(-1,3))
                            
                        pointsscan = pointsscan[~np.isnan(pointsscan).any(axis=1),:]   
                        print('effective points:', pointsscan.shape[0], '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$no_noise')

                        step += 1
                        with open(store_file.replace('.txt', '_'+str(step)+'.txt'),'ab') as f:
                            np.savetxt(f, pointsscan, fmt="%f", delimiter=' ')
                            f.close()
                        with open(store_file_view,'ab') as ff:
                            cc = [[cx,cy,cz]]
                            np.savetxt(ff, cc, fmt="%f", delimiter=' ')
                            ff.close()
                        with open(store_file.replace('.txt', '_split.txt'), "ab") as fff:
                            np.savetxt(fff, [[step, ptt, ptt + pointsscan.shape[0]]], fmt="%d", delimiter=' ')
                            ptt += pointsscan.shape[0]
                            fff.close()
                    if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                        for sp in item.data.vertices:
                            if camera_normal == True:
                                pointsscannoisy.append([sp.co, sp.normal])
                            else:
                                pointsscannoisy.append([sp.co])
                        if camera_normal == True:
                            pointsscannoisy = np.reshape(np.array(pointsscannoisy),(-1,6))
                        else:
                            pointsscannoisy = np.reshape(np.array(pointsscannoisy),(-1,3))
                            
                        pointsscannoisy = pointsscannoisy[~np.isnan(pointsscannoisy).any(axis=1),:]   
                        print('effective points:', pointsscannoisy.shape[0], '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',Noise_sg[i])

                        with open(store_file_N.replace('.txt', '_'+str(Noise_sg[i])+'_'+str(step+1)+'.txt'),'ab') as nf:
                            np.savetxt(nf, pointsscannoisy, fmt="%f", delimiter=' ')
                            nf.close()
                        with open(store_file_N.replace('.txt', '_split.txt'), "ab") as nfff:
                            np.savetxt(nfff, [[step, pttn[i],pttn[i]+pointsscannoisy.shape[0]]], fmt="%d", delimiter=' ')
                            pttn[i] += pointsscannoisy.shape[0]
                            nfff.close()
                del pointsscan
                del pointsscannoisy
            else:
                pointsscannoisy2 = []
                camera.add_scan_mesh = False
                bpy.ops.blensor.scan()
                # store the data of the current scan
                for item in bpy.data.objects:
                    if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                        for sp in item.data.vertices:
                            if camera_normal == True:
                                pointsscannoisy2.append([sp.co, sp.normal])
                            else:
                                pointsscannoisy2.append([sp.co])
                        if camera_normal == True:
                            pointsscannoisy2 = np.reshape(np.array(pointsscannoisy2),(-1,6))
                        else:
                            pointsscannoisy2 = np.reshape(np.array(pointsscannoisy2),(-1,3))
                            
                        pointsscannoisy2 = pointsscannoisy2[~np.isnan(pointsscannoisy2).any(axis=1),:]   
                        print('effective points:', pointsscannoisy2.shape[0], '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',Noise_sg[i] )

                        with open(store_file_N.replace('.txt', '_'+str(Noise_sg[i])+'_'+str(step)+'.txt'),'ab') as nf:
                            np.savetxt(nf, pointsscannoisy2, fmt="%f", delimiter=' ')
                            nf.close()
                del pointsscannoisy2
    bpy.ops.wm.quit_blender()


##################################################################################
# Batch Scan
##################################################################################
# delete default mesh and input the object
bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["Cube"].select = True
bpy.ops.object.delete()

bpy.ops.object.empty_add(type='SPHERE')
bpy.data.objects["Empty"].location=Vector([0.0,0.0,0.0])
bpy.data.objects["Empty"].scale=Vector([0.0,0.0,0.0])

camera_type = ['velodyne', 'ibeo', 'tof', 'kinect', 'generic', 'depthmap']

Source_ = 'meshes'
Out_ = 'Nonuniform'
Out_N = 'Noise'

meshfile = os.path.abspath(sys.argv[-2])
vpsfile = os.path.abspath(sys.argv[-1])

print(str(meshfile), str(vpsfile))

cam_vps = np.loadtxt(vpsfile, delimiter=' ')
Out_file_dir = os.path.join(os.path.dirname(meshfile.replace(Source_, Out_)), os.path.basename(meshfile).replace('.ply',''))
Out_file_dir_N = os.path.join(os.path.dirname(meshfile.replace(Source_, Out_N)), os.path.basename(meshfile).replace('.ply',''))
os.makedirs(Out_file_dir, exist_ok=True)
os.makedirs(Out_file_dir_N, exist_ok=True)
store_file = os.path.join(Out_file_dir, os.path.basename(meshfile).replace('ply','txt'))
store_file_N = os.path.join(Out_file_dir_N, os.path.basename(meshfile).replace('ply','txt'))

Camera_Rotate_Scan(file_loc = meshfile,                             # the ply or other 3d file input dir
                    store_file =  store_file,                      # output dir
                    #sample times
                    store_file_view= os.path.join(Out_file_dir,'viewpoint.txt'),
                    store_file_N = store_file_N,
                    camera_tp = camera_type[3],                    # Camera types, inference the top info 
                    camera_vps = cam_vps,
                    # object scale
                    object_scale = Vector([1.0,1.0,1.0]),          # resize the object
                    # resolution
                    image_x =  512,              
                    image_y =  424,
                    cam_max_dist = 2.1,
                    # Noise parameter
                    Noise_type = 'gaussian',     # ['gaussian','laplace']
                    Noise_mu = 0,                # the average value of the noise
                    Noise_sg = np.array([0.005]),                # the variance valude of the noise 
                    DB_mu = 1,                  
                    DB_sg = 1,
                    Global_Noise_Scale = 0.25,
                    Global_Noise_Smooth = 1.5,
                    camera_normal = False)

bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["Object"].select = True
bpy.ops.object.delete()