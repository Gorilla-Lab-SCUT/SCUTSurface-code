import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import os
import utils
from network import test_net256
import argparse

listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]

##############################################################
# Data Loader
##############################################################
class LoadPatchPairs(Dataset):
    def __init__(self, root, augment=False, repeat=1):
        self.root = root
        self.augment = augment
        self.repeat = repeat
        self.files = listfiles(root)
        self.files = np.tile(np.array(self.files), self.repeat)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        point_patch = self.files[idx]
        point_patch = utils.load_xyz(point_patch)
        
        if bool(random.getrandbits(1)):
            point_patch_pair = point_patch
            cls = 1
        else:
            point_patch_pair_idx = int(np.random.choice(len(self.files), 1))
            if point_patch_pair_idx != idx:
                point_patch_pair = utils.load_xyz(self.files[point_patch_pair_idx])
                cls = 0
            else:
                point_patch_pair = point_patch
                cls = 1
        return {
            'pts1' : point_patch,
            'pts2' : point_patch_pair,
            'cls' : cls,
        }   

##############################################################
# Train Function
##############################################################
def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.CosineSimilarity(dim=1, eps=1e-08)
    train_loss = utils.Averager()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda() 

        pred_1 = model(batch['pts1'])
        pred_2 = model(batch['pts2'])
        gt = batch['cls']

        loss = torch.abs(loss_fn(pred_1, pred_2) - gt).sum()
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_1 = None; pred_2 = None; loss = None
    return train_loss.item()

##############################################################
# Main Function
##############################################################
def main(shape_patch_root, save_path, resume=None, epoch_max = 1000, epoch_save = 100):
    global log, writer

    log, writer = utils.set_save_path(save_path)

    data = LoadPatchPairs(shape_patch_root, repeat=1)
    train_loader = DataLoader(data, batch_size=20, shuffle=True, drop_last=True, num_workers=0)
    model = test_net256(point_dim=6, gf_dim=256).cuda()

    log(model)
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.5)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optimizer_sd'])
        epoch_start = checkpoint['epoch']
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
        print('Resume from', resume)
    else:
        epoch_start = 1

    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        
        sv_file = {
            'model_sd': model_.state_dict(),
            'optimizer_' : 'SGD',
            'optimizer_sd': optimizer.state_dict(),
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--generate_data', action='store_false')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ##############################################################
    # Basic Dir
    ##############################################################
    shape_root = 'Real_GT'
    shape_patch_root = os.path.abspath('RePatch_Scaled')
    os.makedirs(shape_patch_root, exist_ok=True)

    if args.generate_data:
        shape_name = ['bottle_shampoo', 'bowl_chinese', 'cloth_duck', 'coffe_bottle_metal', 'coffe_bottle_plastic', 'cup1', 'flower_pot_2', 'flower_pot', 'gift_box', 'lock', 'marker', 'mouse_two', 'rabbit', 'romoter', 'screwnew', 'tap2', 'toy_cat', 'toy_duck', 'wrench', 'xiaojiejie2']
        # Build Patch Dataset
        utils.build_patch(shape_root, shape_patch_root, shape_name, voxel_size=0.1, point_per_patch=5000, unit_scale=True)

    save_name = args.name
    save_path = os.path.abspath(os.path.join('./save', save_name))
    os.makedirs(save_path, exist_ok=True)

    main(shape_patch_root, save_path, args.resume, epoch_max = 1000, epoch_save = 100)