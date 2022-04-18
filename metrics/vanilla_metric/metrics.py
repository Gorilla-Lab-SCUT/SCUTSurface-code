import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

def eval_pointcloud(pre_mesh_ply, gt_mesh_ply, samplepoint, eval_type, 
                    thresholds=np.linspace(1./1000, 1, 1000)):
        if eval_type == 'real_obj':
            pn_gt = np.loadtxt(gt_mesh_ply, delimiter=' ')
            samplepoint = len(pn_gt)
            pointcloud_tgt = pn_gt[:,:3]
            normals_tgt = pn_gt[:,3:]
            pointcloud, normals = sampleGT(pre_mesh_ply, samplepointsnum = samplepoint)
        else:
            pointcloud_tgt, normals_tgt = sampleGT(gt_mesh_ply, samplepointsnum = samplepoint)
            pointcloud, normals = sampleGT(pre_mesh_ply, samplepointsnum = samplepoint)

        if pointcloud.shape[0] == 0 or pointcloud_tgt.shape[0] == 0:
            out_dict = {
                'N_Acc' : 0,
                'N_Comp' : 0,
                'normals': 0,
                'CD_Acc' : 0,
                'CD_Comp': 0,
                'chamfer-L2': 0,
                'F_Acc_005': 0,
                'F_Comp_005': 0,
                'f-score-005': 0, # threshold = 0.005  for syn_obj
                'F_Acc_03': 0,
                'F_Comp_03': 0,
                'f-score-03': 0,  # threshold = 0.03    for syn_scene
                'F_Acc_5': 0,
                'F_Comp_5': 0,
                'f-score-5': 0,   # threshold = 0.5      for real_obj
            }
            return out_dict

        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL2 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]

        out_dict = {
            'N_Acc' : accuracy_normals,
            'N_Comp' : completeness_normals,
            'normals': normals_correctness,
            'CD_Acc' : accuracy,
            'CD_Comp': completeness,
            'chamfer-L2': chamferL2,
            'F_Acc_005': precision[4],
            'F_Comp_005': recall[4],
            'f-score-005': F[4],       # threshold = 0.005  for syn_obj
            'F_Acc_03': precision[29],
            'F_Comp_03': recall[29],
            'f-score-03': F[29],       # threshold = 0.03    for syn_scene
            'F_Acc_5': precision[499],
            'F_Comp_5': recall[499],
            'f-score-5': F[499],        # threshold = 0.5      for real_obj
        }
        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src, p=2, k=1, n_jobs=8)    # p=2, Euclidean distance

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def sampleGT(filename, samplepointsnum):
    ''' Sample a mesh, return points with normals.
    Args:
        filename (str): input mesh file
        samplepointsnum (int): numbers of points to be sampled
    '''
    mesh = trimesh.load(filename)
    mesh.fix_normals()  # ensuring outward normals
    sample_random, index_random = trimesh.sample.sample_surface(mesh, samplepointsnum)
    sample_normal = mesh.face_normals[index_random]
    return sample_random, sample_normal