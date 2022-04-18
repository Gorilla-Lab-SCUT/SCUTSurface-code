import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(__file__))
import utils
from network import test_net256
import torch.nn.functional as F
import argparse
from tqdm import tqdm


def get_paris(pred_ply, gt_xyz, voxel_size=0.1, unit_scale=True, eval_type='real_obj'):
    if eval_type == 'real_obj':
        gt_xyzn = utils.load_xyz(gt_xyz)
        print('++++++++++loaded gt+++++++++++++')
        pointcloud, normals = utils.sampleGT(pred_ply, samplepointsnum = len(gt_xyzn))
        pred_xyzn = np.concatenate((pointcloud, normals), axis=1)
        print('----------loaded pred----------')
    else:
        gt_p, gt_n = utils.sampleGT(gt_xyz, samplepointsnum= 1000000)
        gt_xyzn = np.concatenate((gt_p, gt_n), axis=1)
        print('++++++++++loaded gt+++++++++++++')
        pointcloud, normals = utils.sampleGT(pred_ply, samplepointsnum = len(gt_xyzn))
        pred_xyzn = np.concatenate((pointcloud, normals), axis=1)
        print('----------loaded pred----------')

    if unit_scale:
        scale = np.abs(gt_xyzn[:,:3]).max()
        gt_xyzn[:,:3] = gt_xyzn[:,:3] / scale
        pred_xyzn[:,:3] = pred_xyzn[:,:3] / scale

    pred_gt_paris = utils.voxelize_for_test_overlap(pred_xyzn, gt_xyzn, voxel_size)
    return pred_gt_paris


def main(model_path, pred_ply, gt_xyz, voxel_size=0.1, point_per_patch=8000, eval_type='real_obj'):
    print('Voxel Size is : ', voxel_size, '\t', 'Point Per Patch is : ', point_per_patch)
    test_pair = get_paris(pred_ply, gt_xyz, voxel_size, unit_scale=True, eval_type=eval_type)

    # model = test_net(point_dim=6, gf_dim=128).cuda()
    model = test_net256(point_dim=6, gf_dim=256).cuda()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_sd'])
    model.eval()
    # print(model)

    Similarities = []
    for batch in tqdm(test_pair, leave=False, desc='Testing'):
        pred_pts = batch['pred_pts']
        gt_pts = batch['gt_pts']
        # print(batch)
        if (len(pred_pts) == 0) or (len(gt_pts) == 0):
            Similarities.append(0.0)
        else:
            pred_choice = np.random.choice(len(pred_pts), point_per_patch, replace=True)
            pred_pts = pred_pts[pred_choice]
            gt_choice = np.random.choice(len(gt_pts), point_per_patch, replace=True)
            gt_pts = gt_pts[gt_choice]
            pred_pts_tensor = torch.FloatTensor(pred_pts).view(1, -1, 6).cuda()
            gt_pts_tensor = torch.FloatTensor(gt_pts).view(1, -1, 6).cuda()

            feat_pred = model(pred_pts_tensor)
            feat_gt = model(gt_pts_tensor)

            score = torch.abs(F.cosine_similarity(feat_pred.view(1,-1), feat_gt.view(1,-1))).sum()
            Similarities.append(score.detach().cpu().numpy())
    Neural_Socre = np.sum(np.array(Similarities))
    print(Neural_Socre/len(test_pair))
    return Neural_Socre/len(test_pair)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', default='0')
    argparser.add_argument("--eval_type", "-et", type=str, metavar=['syn_obj', 'syn_scene', 'real_obj'], 
                           default='syn_obj', help="which dataset to eval")
    argparser.add_argument("--in_dir", "-pr", type=str, default=None,
                           help="in file dir")
    argparser.add_argument("--gt_dir", "-gt", type=str, default='/mnt/h/srcclassify_five/GTMesh/',
                           help="gt file dir")
    argparser.add_argument("--model_dir", "-md", type=str, default='save/T197_Scaled_net256_Adam/epoch-1000.pth',
                           help="model file dir")
    argparser.add_argument("--voxel_size", "-vs", type=float, default=0.1,
                           help="voxel size")
    argparser.add_argument("--point_per_patch", "-ppp", type=int, default=5000,
                           help="point_per_patch")
    argparser.add_argument("--out_csv", "-csv", type=str, default='test.csv',
                           help="result file, store in csv type")
    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    
    print(args)

    replaceword = ['Nonuniform', 'Uniform', 'Noise1','Noise2','Noise3',
                    'Outlier1', 'Outlier2', 'Outlier3', 
                    'Missing_Data1','Missing_Data2','Missing_Data3',
                    'Misalignment1','Misalignment2', 'Misalignment3']

    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    files = listfiles(args.in_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    for i in tqdm(range(len(files))):
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
            NotImplementedError
        print('Eval on : ', args.eval_type, 'dataset \t', files[i], '----> ', gt_mesh_file)
        pred_ply = os.path.abspath(files[i]).replace('\\','/').replace('//','/')
        gt_xyz = os.path.abspath(gt_mesh_file).replace('\\','/').replace('//','/')
        score = main(args.model_dir, pred_ply, gt_xyz, args.voxel_size, args.point_per_patch, args.eval_type)
        out_dict = {
            'filename' : pred_ply.split('/')[-4]+'/'+pred_ply.split('/')[-3]+'/'+pred_ply.split('/')[-2]+'/'+pred_ply.split('/')[-1],
            'gtfile': gt_xyz.split('/')[-4]+'/'+gt_xyz.split('/')[-3]+'/'+gt_xyz.split('/')[-2]+'/'+gt_xyz.split('/')[-1],
            'score' : score,
        }
        utils.writecsv(args.out_csv, out_dict)