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

def get_label_path(path):
    pass

def load_semantic_label(sequence,frame):
    label_path=args.data_root+'/'+sequence+'/label_2/'+frame+'.npy'
    label=np.load(label_path)

def eval():
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']
    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    
    transform=transforms.SemanticKittiProcessData(args.data_process,args.num_points,args.allow_less_points)
    val_dataset=datasets.SemanticKitti(train=False,test=False,transform=transform,num_points=args.num_points,data_root=args.data_root)
    collate=None
    if args.data_process['DOWN_SAMPLE_METHOD']=='voxel':
        collate=datasets.Collater(args.data_process)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,collate_fn=collate)
    
    model = PointConvSceneFlow()

    pretrain=args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)

    model = model.cuda()

    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        if args.data_process['DOWN_SAMPLE_METHOD']=='random':
                pos1, pos2, norm1, norm2, flow, batch_path = data
        elif args.data_process['DOWN_SAMPLE_METHOD']=='voxel':
            pos1, pos2, norm1, norm2, flow, batch_path,pos1_mask, pos2_mask, norm1_mask, norm2_mask, flow_mask = data 

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
                sequence, frame = get_label_path(batch_path[i])
                semantic_label=load_semantic_label(sequence,frame)

                total_mean_epe3d=torch.norm(full_flow-flow,dim=2).mean()
                # static_epe3d

                # moving_epe3d
