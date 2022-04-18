import os
import numpy as np
from tqdm import tqdm
import argparse
from utils import load_config, ssubprocess
import concurrent.futures

def main(cfg):
    main = cfg['cgal_prepoce_jet']['main']
    in_dir = cfg['cgal_prepoce_jet']['in_dir']
    out_dir = cfg['cgal_prepoce_jet']['out_dir']

    os.makedirs(out_dir,exist_ok=True)

    choosefun = cfg['cgal_prepoce_jet']['choosefun']  # 0-Remove Outlier /1-Redistration/2-Jet Smoothing/3-Bilateral Smoothing
    Argv4 = cfg['cgal_prepoce_jet']['Argv4']

    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    files = listfiles(in_dir)

    cmds = []
    for fileone in tqdm(files):
      if fileone.split('.')[-1] == 'txt':
        outfile = fileone.replace(in_dir, out_dir)
        # Jet Smooth
        exe = str(main+' '+str(fileone)+' '+str(outfile)+' '+str(choosefun)+' '+str(Argv4))
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
        cmds.append(exe)

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg['num_worker'])) as executor:
        for i in range(len(cmds)):
            string = str(i) +'/'+ str(len(cmds))
            executor.submit(ssubprocess, string,cmds[i])
        executor.shutdown()
    print("##############################################################################################")
    for fileone in tqdm(files):
        if fileone.split('.')[-1] == 'txt':
          print(fileone)
          outfile = fileone.replace(in_dir, out_dir)
          normals = np.loadtxt(fileone, delimiter=' ')[:,3:]
          pts = np.loadtxt(outfile, delimiter=' ')[:,:3]
          ptsn = np.concatenate((pts,normals),axis=-1)
          np.savetxt(outfile,ptsn, delimiter=' ', fmt='%f')
    print("CGAL Smooth finished!")
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='!!!De-noise!!!')
    parser.add_argument('--configure', '-conf', type=str, default='config_example.yaml', help='the config yaml file')
    args = parser.parse_args()
    cfg = load_config(str(args.configure))
    main(cfg)
    