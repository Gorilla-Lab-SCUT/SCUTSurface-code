import numpy as np
import os
import time
from tqdm import tqdm
import data_utils

########################################################################
# Get Uniform Data  fps
########################################################################
def Uniform_one(fileidx_file, ptsssss_file, viewpoint_file, concate_file, num_points, FPS, string=None):
    print(string)
    fileidx = np.loadtxt(fileidx_file, delimiter=' ')
    ptsssss = np.loadtxt(ptsssss_file, delimiter=' ')
    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = np.arange(len(viewpoint))
    with open(concate_file, 'ab') as f:
        for idx in Index:
            np.savetxt(f, ptsssss[int(fileidx[idx][1]):int(fileidx[idx][2])] , fmt="%f",delimiter=' ')
    f.close()
    del ptsssss
    if FPS:
        print('Conduct FPS ----> ', concate_file)
        starttime = time.time()
        mispoint =  np.loadtxt(concate_file, delimiter=' ')
        mispoint = data_utils.farthest_point_sample(mispoint, int(num_points))
        np.savetxt(concate_file, mispoint, fmt = "%f" ,delimiter=' ')
        del mispoint
        print(concate_file, '----> FPS Time: ', time.time() - starttime)


def Get_Uniform(In_Dir, Out_Dir, FPS=True, num_work=1):
    print("Get Uniform Data")
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    
    cmds = []
    i = 0
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        num_points = data_utils.howmuchpoint(pathsmc, 0)
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
            concate_file = os.path.join(Out_Dir, pathsmc, str(pathobj)+ ".txt")
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ ".txt")
            viewpoint_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,"viewpoint.txt")
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file, num_points, FPS, string))
            i += 1
    data_utils.start_process_pool(Uniform_one, cmds, num_work)


########################################################################
# Get Nonuniform Data   random sample
########################################################################
def Nonuniform_one(fileidx_file, ptsssss_file, viewpoint_file, concate_file, num_points, string=None):
    print(string)
    fileidx = np.loadtxt(fileidx_file, delimiter=' ')
    ptsssss = np.loadtxt(ptsssss_file, delimiter=' ')
    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = np.arange(len(viewpoint))
    with open(concate_file, 'ab') as f:
        for idx in Index:
            np.savetxt(f, ptsssss[int(fileidx[idx][1]):int(fileidx[idx][2])] , fmt="%f",delimiter=' ')
    f.close()
    del ptsssss
    # print(concate_file)
    starttime = time.time()
    mispoint =  np.loadtxt(concate_file, delimiter=' ')
    mispoint = data_utils.randomsample(mispoint, int(num_points))
    np.savetxt(concate_file, mispoint, fmt = "%f" ,delimiter=' ')
    del mispoint
    print(concate_file, '----> RS Time: ', time.time() - starttime)

def Get_Noniform(In_Dir, Out_Dir, num_work=1):
    print("Get Nonuniform Data")
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    
    cmds = []
    i=0
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        num_points = data_utils.howmuchpoint(pathsmc, 0)
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
            concate_file = os.path.join(Out_Dir, pathsmc, str(pathobj)+ ".txt")
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ ".txt")
            viewpoint_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,"viewpoint.txt")
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file, num_points, string))
            i += 1
    data_utils.start_process_pool(Nonuniform_one, cmds, num_work)


########################################################################
# Get Noise Data   fps
########################################################################
def Noise_one(fileidx_file, ptsssss_file, viewpoint_file, concate_file, num_points, FPS, string=None):
    print(string)
    fileidx = np.loadtxt(fileidx_file, delimiter=' ')
    ptsssss = np.loadtxt(ptsssss_file, delimiter=' ')

    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = np.arange(len(viewpoint))
    with open(concate_file, 'ab') as f:
        for idx in Index:
            np.savetxt(f, ptsssss[int(fileidx[idx][1]):int(fileidx[idx][2])] , fmt="%f",delimiter=' ')
    f.close()
    del ptsssss
    starttime = time.time()
    if FPS:
        print('Conduct FPS ----> ', concate_file)
        mispoint =  np.loadtxt(concate_file, delimiter=' ')
        mispoint = data_utils.farthest_point_sample(mispoint, int(num_points))
        np.savetxt(concate_file, mispoint, fmt = "%f" ,delimiter=' ')
        del mispoint
    print(concate_file, '----> FPS Time: ', time.time() - starttime)

def Get_Noise(In_Dir, In_DirV, Out_Dir, anoise, FPS=False, num_work=1):
    print("Get Noise Data")
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    cmds = []
    i=0
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        num_points = data_utils.howmuchpoint(pathsmc, 0)
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
            concate_file = os.path.join(Out_Dir, pathsmc, str(pathobj)+ ".txt")
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_"+str(anoise)+".txt")
            viewpoint_file = os.path.join(In_DirV, pathsmc, str(pathobj) ,"viewpoint.txt")
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file, num_points, FPS, string))
            i += 1
    data_utils.start_process_pool(Noise_one, cmds, num_work)


########################################################################
# Get Outlier Data
########################################################################
def Outlier_one(fileidx_file, ptsssss_file, viewpoint_file, concate_file, outlier_number, intensity, num_points, FPS=True, string=None):
    fileidx = np.loadtxt(fileidx_file, delimiter=' ')
    ptsssss = np.loadtxt(ptsssss_file, delimiter=' ')
    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = np.arange(len(viewpoint))
    with open(concate_file, 'ab') as f:
        for idx in Index:
            pts = ptsssss[int(fileidx[idx][1]):int(fileidx[idx][2])]
            ptsoutlier = data_utils.outlierbpoints(pts, number = outlier_number, intensity = intensity)
            pts = np.concatenate((pts,ptsoutlier), axis = 0)
            np.savetxt(f, pts, fmt="%f",delimiter=' ')
    f.close()
    del ptsssss
    if FPS:
        print('Conduct FPS ----> ', concate_file)
        starttime = time.time()
        mispoint =  np.loadtxt(concate_file, delimiter=' ')
        mispoint = data_utils.farthest_point_sample(mispoint, int(num_points))
        np.savetxt(concate_file, mispoint, fmt = "%f" ,delimiter=' ')
        del mispoint
        print(concate_file, '----> FPS Time: ', time.time() - starttime)

def Get_outlier(In_Dir, Out_Dir, number, intensity = 0.15, FPS=True, num_work=1):
    print("Get Outlier Data")
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    cmds = []
    i = 0
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        num_points = data_utils.howmuchpoint(pathsmc, 0)
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
            concate_file = os.path.join(Out_Dir, pathsmc, str(pathobj)+ ".txt")
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ ".txt")
            viewpoint_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,"viewpoint.txt")
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file, number, intensity, num_points, FPS, string))
            i += 1
    data_utils.start_process_pool(Outlier_one, cmds, num_work)


########################################################################
# Get Missing Data   fps
########################################################################
def Missingdata_one(fileidx_file, ptsssss_file, viewpoint_file, concate_file, zanglelist, num_points, FPS=False, string=None):
    print(string)
    fileidx = np.loadtxt(fileidx_file, delimiter=' ')
    ptsssss = np.loadtxt(ptsssss_file, delimiter=' ')
    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = data_utils.discardvpsbytracket(viewpoint, zanglelist)
    with open(concate_file, 'ab') as f:
        for idx in Index:
            np.savetxt(f, ptsssss[int(fileidx[idx][1]):int(fileidx[idx][2])] , fmt="%f",delimiter=' ')
    f.close()
    del ptsssss
    if FPS:
        print('Conduct FPS ----> ', concate_file)
        starttime = time.time()
        mispoint =  np.loadtxt(concate_file, delimiter=' ')
        if len(mispoint) > num_points:
            mispoint = data_utils.farthest_point_sample(mispoint, int(num_points))
        np.savetxt(concate_file, mispoint, fmt = "%f" ,delimiter=' ')
        del mispoint
        print(concate_file, '----> FPS Time: ', time.time() - starttime)

def Get_MissingData(In_Dir, Out_Dir, zanglelist, FPS=False, num_work=1):
    print("Get Missing Data ")
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    cmds = []
    i=0
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        num_points = data_utils.howmuchpoint(pathsmc, 0)
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
            concate_file = os.path.join(Out_Dir, pathsmc, str(pathobj)+ ".txt")
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ ".txt")
            viewpoint_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,"viewpoint.txt")
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file, zanglelist, num_points, FPS, string))
            i += 1
    data_utils.start_process_pool(Missingdata_one, cmds, num_work)     


########################################################################
# Get Misalignment Data   fps
########################################################################
def Misalignment_one(fileidx_file, ptsssss_file, viewpoint_file, concate_file, angle, intensity, num_points, FPS=False, string=None):
    print(string)
    fileidx = np.loadtxt(fileidx_file, delimiter=' ')
    ptsssss = np.loadtxt(ptsssss_file, delimiter=' ')
    viewpoint = np.loadtxt(viewpoint_file, delimiter=' ')
    Index = len(viewpoint)
    Filechose = np.random.choice(Index, int(0.5 * Index), replace=False)
    num_count = 0
    with open(concate_file, 'ab') as f:
        for idx in Filechose:
            ptsone = ptsssss[int(fileidx[idx][1]):int(fileidx[idx][2])]
            ptsone = data_utils.farthest_point_sample(ptsone, ptsone.shape[0] // 4)
            ptsone = data_utils.disturbpoints(ptsone, angle, intensity)
            # ptsone = ptsone[~np.isnan(ptsone).any(axis=1),:]   #删掉nan
            np.savetxt(f, ptsone, fmt="%f", delimiter=' ')
            del ptsone    
            num_count += ((int(fileidx[idx][2]) - int(fileidx[idx][1]))//4)
            # num_count += ((int(fileidx[idx][2]) - int(fileidx[idx][1])))
            if num_count > num_points * 4 :
                break  
    f.close()
    del ptsssss
    if FPS:
        print('Conduct FPS ----> ', concate_file)
        starttime = time.time()
        mispoint =  np.loadtxt(concate_file, delimiter=' ')
        if len(mispoint) > num_points:
            mispoint = data_utils.farthest_point_sample(mispoint, int(num_points))
        np.savetxt(concate_file, mispoint, delimiter=' ')
        del mispoint
        print(concate_file, '----> FPS Time: ', time.time() - starttime)


def Get_Missalignment(In_Dir, Out_Dir,intensity, angle=0, FPS=False, num_work=1):
    print("Get Misalignment Data")
    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    fileslen = len(listfiles(In_Dir))
    cmds = []
    i=0
    for pathsmc in tqdm(data_utils.path_name(In_Dir)):
        num_points = data_utils.howmuchpoint(pathsmc, 0)
        for pathobj in data_utils.path_name(os.path.join(In_Dir, pathsmc)):
            listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
            concate_file = os.path.join(Out_Dir, pathsmc, str(pathobj)+ ".txt")
            os.makedirs(os.path.dirname(concate_file),exist_ok=True)
            fileidx_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ "_split.txt")
            ptsssss_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,str(pathobj)+ ".txt")
            viewpoint_file = os.path.join(In_Dir, pathsmc, str(pathobj) ,"viewpoint.txt")
            string = str(i) +'/'+ str(fileslen) + ' -----> ' + ptsssss_file
            cmds.append((fileidx_file, ptsssss_file, viewpoint_file, concate_file, angle, intensity, num_points, FPS, string))
    data_utils.start_process_pool(Misalignment_one, cmds, num_work)