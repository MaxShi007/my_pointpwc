import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d
from torchsparse.utils.quantize import sparse_quantize


def get_label_path(path):
    filename=os.path.basename(path)
    sequence, frame = filename.split('_')
    return sequence, frame

def load_semantic_label(sequence,frame):
    global label_root
    label_path=os.path.join(label_root,"sequences",sequence,"labels",frame.replace('npz',"label"))
    label=np.fromfile(label_path,dtype=np.uint32).reshape(-1)
    sem_label = label & 0xFFFF
    ins_label = label >> 16
    return sem_label

def load_non_ground_mask(sequence,frame):
    global non_ground_root
    ground_label=os.path.join(non_ground_root,sequence,"ground_labels",frame.replace('npz',"label"))
    label=np.fromfile(ground_label,dtype=np.uint32).reshape(-1)
    sem_label = label & 0xFFFF
    non_ground_mask=(sem_label!=9)
    ground_mask=(sem_label==9)
    return non_ground_mask

def get_voxel_downsamping_indices(path,voxel_size):
    
    data=np.load(path)
    current_point=data['current_point']
    changed_current_point=current_point.copy()
    min=np.min(current_point,axis=0,keepdims=True)
    changed_current_point-=min

    coords,indices,inverse=sparse_quantize(changed_current_point,voxel_size=voxel_size,return_index=True,return_inverse=True)
    return indices


def eval():
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']
    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    
    transform=transforms.SemanticKittiProcessData(args.data_process,args.num_points,args.allow_less_points)
    val_dataset=datasets.SemanticKitti(train=False,test=True,transform=transform,num_points=args.num_points,data_root=args.data_root)
    collate=None
    if args.data_process['DOWN_SAMPLE_METHOD']=='voxel':
        collate=datasets.Collater(args.data_process)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,collate_fn=collate)
    
    model = PointConvSceneFlow()

    pretrain=args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)

    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True 
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model = model.cuda()

    static_epe3ds=AverageMeter()
    moving_epe3ds=AverageMeter()
    total_epe3ds=AverageMeter()
    acc3d_stricts=AverageMeter()
    acc3d_relaxs=AverageMeter()
    outliers=AverageMeter()
    moving_acc3d_stricts=AverageMeter()
    moving_acc3d_relaxs=AverageMeter()
    moving_outliers=AverageMeter()

    sequences=set()

    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        if args.data_process['DOWN_SAMPLE_METHOD']=='random':
                pos1, pos2, norm1, norm2, flow, batch_path,_,_,_ = data
        elif args.data_process['DOWN_SAMPLE_METHOD']=='voxel':
            pos1, pos2, norm1, norm2, flow, batch_path,pos1_mask, pos2_mask, norm1_mask, norm2_mask, flow_mask ,_,_,_= data 

        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        model = model.eval()
        with torch.no_grad(): 
            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

            full_flow = pred_flows[0].permute(0, 2, 1)
            pos1=pos1.cpu().numpy()
            pos2=pos2.cpu().numpy()
            pred_flow=full_flow.cpu().numpy()
            flow_gt=flow.cpu().numpy()

            for i in range(len(batch_path)):
                print(batch_path[i])
                current_indices=get_voxel_downsamping_indices(batch_path[i],voxel_size=args.data_process['VOXEL_SIZE'])
                sequence, frame = get_label_path(batch_path[i])
                sequences.add(sequence)
                semantic_label=load_semantic_label(sequence,frame)
                non_ground_mask=load_non_ground_mask(sequence,frame)
                non_ground_semantic_label=semantic_label[non_ground_mask]

                non_ground_ds_semantic_label=non_ground_semantic_label[current_indices]
                moving_mask=non_ground_ds_semantic_label>250
                static_mask=non_ground_ds_semantic_label<=250

                pos1_mask_i=pos1_mask[i].astype(np.bool)
                flow_mask_i=flow_mask[i].astype(np.bool)

                pos1_i=pos1[i][pos1_mask_i]
                pred_flow_i=pred_flow[i][flow_mask_i]
                flow_gt_i=flow_gt[i][flow_mask_i]

                moving_pos1=pos1_i[moving_mask]
                static_pos1=pos1_i[static_mask]
                moving_pred_flow=pred_flow_i[moving_mask]
                static_pred_flow=pred_flow_i[static_mask]
                moving_flow_gt=flow_gt_i[moving_mask]
                static_flow_gt=flow_gt_i[static_mask]

                # static_epe3d
                static_l2_norm=np.linalg.norm(static_pred_flow-static_flow_gt,axis=1)
                static_epe3d=static_l2_norm.mean()
                # static_gt_norm=np.linalg.norm(static_flow_gt, axis=-1)
                # static_relative_err = static_l2_norm / (static_gt_norm + 1e-4)
                # static

                # moving_epe3d
                if len(moving_pred_flow)!=0:
                    moving_l2_norm=np.linalg.norm(moving_pred_flow-moving_flow_gt,axis=1)
                    moving_epe3d=moving_l2_norm.mean()
                    moving_epe3ds.update(moving_epe3d)

                    gt_norm=np.linalg.norm(moving_flow_gt, axis=-1)
                    relative_err=moving_l2_norm/(gt_norm + 1e-4)
                    moving_acc3d_strict=(np.logical_or(moving_l2_norm<0.05,relative_err<0.05)).astype(np.float).mean()
                    moving_acc3d_relax=(np.logical_or(moving_l2_norm<0.1,relative_err<0.1)).astype(np.float).mean()
                    moving_outlier=(np.logical_or(moving_l2_norm>0.3,relative_err>0.1)).astype(np.float).mean()
                    moving_acc3d_stricts.update(moving_acc3d_strict)
                    moving_acc3d_relaxs.update(moving_acc3d_relax)
                    moving_outliers.update(moving_outlier)
                # total_epe3d
                total_epe3d,acc3d_strict,acc3d_relax,outlier=evaluate_3d(pred_flow_i,flow_gt_i)

                static_epe3ds.update(static_epe3d)
                total_epe3ds.update(total_epe3d)
                acc3d_stricts.update(acc3d_strict)
                acc3d_relaxs.update(acc3d_relax)
                outliers.update(outlier)
    res_str=('sequence:{sequence_}\n'
            'static_epe3d:{static_epe3d_.avg:.4f}\n'
            'moving_epe3d:{moving_epe3d_.avg:.4f}\n'
            'total_epe3d:{total_epe3d_.avg:.4f}\n'

            'moving_acc3d_strict:{moving_acc3d_strict_.avg:.4f}\n'
            'moving_acc3d_relax:{moving_acc3d_relax_.avg:.4f}\n'
            'moving_outlier:{moving_outlier_.avg:.4f}\n'

            'acc3d_strict:{acc3d_strict_.avg:.4f}\n'
            'acc3d_relax:{acc3d_relax_.avg:.4f}\n'
            'outlier:{outlier_.avg:.4f}\n'.format(moving_outlier_=moving_outliers,moving_acc3d_relax_=moving_acc3d_relaxs,moving_acc3d_strict_=moving_acc3d_stricts,sequence_=sequences,static_epe3d_=static_epe3ds,moving_epe3d_=moving_epe3ds,total_epe3d_=total_epe3ds,acc3d_strict_=acc3d_stricts,acc3d_relax_=acc3d_relaxs,outlier_=outliers))
    print('*'*50)
    print(res_str)
    print('*'*50)

if __name__=='__main__':
    ##############
    # 该eval仅评估下采样后的点，不评估网络估计的flow上采样后的结果，因为下采样后的点是网络直接输出的，可以用来评判网络性能
    #################
    label_root='/share/sgb/semantic_kitti/dataset'
    non_ground_root='/share/sgb/kitti-ground'
    eval()
