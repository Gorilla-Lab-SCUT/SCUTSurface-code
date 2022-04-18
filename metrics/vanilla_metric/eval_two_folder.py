import os
import argparse
import pandas as pd
from metrics import eval_pointcloud
import multiprocessing
import subprocess

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
        subprocess.run(call)
    else:
        #subprocess.run(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # suppress outputs
        subprocess.run(call, stdout=subprocess.DEVNULL)

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


def eval_pip(pre_mesh_file, gt_mesh_file, samplepoint, csvfile, eval_type, string):
    print(string)
    pre_mesh_file = os.path.abspath(pre_mesh_file)
    gt_mesh_file = os.path.abspath(gt_mesh_file)
    pre_mesh_file = pre_mesh_file.replace('\\','/').replace('//','/')
    gt_mesh_file = gt_mesh_file.replace('\\','/').replace('//','/')
    out_dict = {
                'filename' : pre_mesh_file.split('/')[-4]+'/'+pre_mesh_file.split('/')[-3]+'/'+pre_mesh_file.split('/')[-2]+'/'+pre_mesh_file.split('/')[-1],
                'gtfile': gt_mesh_file.split('/')[-4]+'/'+gt_mesh_file.split('/')[-3]+'/'+gt_mesh_file.split('/')[-2]+'/'+gt_mesh_file.split('/')[-1],
            }
    eval_dict = eval_pointcloud(pre_mesh_file, gt_mesh_file, samplepoint, eval_type)
    out_dict = {**out_dict, **eval_dict}
    writecsv(csvfile, out_dict)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eval_type", "-et", type=str, metavar=['syn_obj', 'syn_scene', 'real_obj'], 
                           default='syn_obj', help="which dataset to eval")
    argparser.add_argument("--in_dir", "-pr", type=str, default=None,
                           help="in file dir")
    argparser.add_argument("--gt_dir", "-gt", type=str, default='/mnt/h/srcclassify_five/GTMesh/',
                           help="gt file dir")
    argparser.add_argument("--samplepoints", "-sp", type=int, default=200000,
                           help="sample point numbers")
    argparser.add_argument("--out_csv", "-csv", type=str, default='testspsr.csv',
                           help="result file, store in csv type")
    argparser.add_argument("--num_worker", "-nw", type=int, default=4,
                            help="the num_work for multiprocessing")
    args = argparser.parse_args()
    
    print(args)

    replaceword = ['Nonuniform', 'Uniform', 'Noise1','Noise2','Noise3',
                    'Outlier1', 'Outlier2', 'Outlier3', 
                    'Missing_Data1','Missing_Data2','Missing_Data3',
                    'Misalignment1','Misalignment2', 'Misalignment3']

    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    files = listfiles(args.in_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    
    cmds = []
    for i in range(len(files)):
        if args.eval_type == 'syn_obj':
            gt_mesh_file = files[i].replace(args.in_dir, args.gt_dir)
            for j in range(len(replaceword)):
                gt_mesh_file = gt_mesh_file.replace(replaceword[j], '')
        elif args.eval_type == 'syn_scene':
            gt_mesh_file = os.path.join(args.gt_dir, os.path.basename(files[i]))
            gt_mesh_file = gt_mesh_file.replace('_0.003_pcs','').replace('_0.005_pcs','')
        elif args.eval_type == 'real_obj':
            gt_mesh_file = os.path.join(args.gt_dir, os.path.basename(files[i]))
            gt_mesh_file = gt_mesh_file.replace('_pcd.ply','.xyz')
        else:
            print("Error Datasets to eval")
            return
        print('Eval on : ', args.eval_type, 'dataset \t', files[i], '----> ', gt_mesh_file)
        string = str(i) +'/'+ str(len(files)) + ' -----> ' + files[i] +'##'+ gt_mesh_file
        cmds.append((files[i], gt_mesh_file, args.samplepoints, args.out_csv, args.eval_type, string))
    start_process_pool(eval_pip, cmds, args.num_worker)

if __name__ == "__main__":
    main()