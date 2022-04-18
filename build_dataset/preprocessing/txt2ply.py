import os
import numpy as np
from tqdm import tqdm
import argparse

def txt2ply(txtfile, plyfile):
    points = np.loadtxt(txtfile, delimiter=' ')
    np.savetxt(plyfile, points, fmt="%f", delimiter=' ')
    strr = ("ply \nformat ascii 1.0 \ncomment github.com/mikedh/trimesh \nelement vertex %d \nproperty float x \nproperty float y \nproperty float z \nproperty float nx \nproperty float ny \nproperty float nz \nelement face 0 \nproperty list uchar int vertex_indices \nend_header\n" %(len(points)))

    with open(plyfile, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(strr)
        f.write(old)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='!!! Conver txt to ply !!!')
    parser.add_argument('--In_txt_Dir', type=str, default='None')
    parser.add_argument('--Out_ply_Dir', type=str, default='None')
    args = parser.parse_args()

    in_txt_dir= os.path.abspath(args.In_txt_Dir)
    out_ply_dir= os.path.abspath(args.Out_ply_Dir)

    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    files = listfiles(in_txt_dir)
    for fileone in tqdm(files):
        if fileone.split('.')[-1] == 'txt':
            filedir2 = fileone.replace(in_txt_dir, out_ply_dir).replace('txt','ply')
            os.makedirs(os.path.dirname(filedir2),exist_ok=True)
            txt2ply(fileone, filedir2)