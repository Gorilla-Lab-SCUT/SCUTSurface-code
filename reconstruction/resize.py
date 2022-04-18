import os
import argparse
import numpy as np
from scipy.sparse.sputils import validateaxis
import trimesh
from plyfile import PlyData
listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]


def _normalize_mesh(file_in, file_out, scale, trans=None):
    """resize mesh file.
    Args:
      file_in: str, filename for mesh file to load
      file_out: str, filename for mesh file to save
      scale : float, 
      trans : [x,y,z] float, 
    """
    mesh = trimesh.load(file_in)
    # scale 
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    # translate 
    if trans is not None:
        translation = trimesh.transformations.translation_matrix(direction=trans)
        mesh.apply_transform(translation)
    mesh.export(file_out)

def _normalize_mesh_by_vertice(file_in, file_out, scale, trans=None):
    mesh = trimesh.load(file_in)
    vertice = mesh.vertices
    face = mesh.faces
    # scale
    vertice = vertice * scale
    # translate 
    if trans is not None:
        vertice = vertice + trans
    new_mesh = trimesh.Trimesh(vertice, face)
    new_mesh.export(file_out)

def read_point_ply(filename):
    """Load point cloud from ply file.
    Args:
      filename: str, filename for ply file to load.
    Returns:
      v: np.array of shape [#v, 3], vertex coordinates
    """
    pd = PlyData.read(filename)['vertex']
    try:
        v = np.array(np.stack([pd[i] for i in ['x', 'y', 'z']], axis=-1))
    except:
        v = np.array(np.stack([pd[i] for i in ['x', 'y', 'z']], axis=-1))
    return v

def resize(In_Dir,In_PC_Dir, Out_Dir, scales_type):
    os.makedirs(Out_Dir, exist_ok=True)
    files = listfiles(In_Dir)
    for fileone in files:
        outfile = fileone.replace(In_Dir, Out_Dir)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        pc_file = fileone.replace(In_Dir, In_PC_Dir)
        print(fileone, '--' ,outfile, '--', pc_file)
        
        xyz = read_point_ply(pc_file)
        xyz = xyz[:,:3]
        trans = xyz.mean(0)
        if scales_type == 'SAL' or scales_type == 'IGR' or scales_type=='LIG_Tensorflow':
            scale = np.abs(xyz-trans).max()
        elif scales_type == 'LIG_Pytorch':
            scale = np.linalg.norm((xyz-trans), ord=2, axis=1).max() + 1e-12
        elif scales_type == 'Points2Surf':
            scale = np.abs((xyz-trans)).max() * 2.1
        else:
            NotImplementedError
        _normalize_mesh(fileone, outfile, scale, trans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--In_PC_Dir', type=str, default=None)
    parser.add_argument('--In_Dir', type=str, default=None)
    parser.add_argument('--Out_Dir', type=str, default=None)
    parser.add_argument('--scales_type', type=str, default=None)
    args = parser.parse_args()

    In_PC_Dir = os.path.abspath(args.In_PC_Dir)
    In_Dir = os.path.abspath(args.In_Dir)
    Out_Dir = os.path.abspath(args.Out_Dir)
    scales_type = args.scales_type

    resize(In_Dir, In_PC_Dir, Out_Dir, scales_type)