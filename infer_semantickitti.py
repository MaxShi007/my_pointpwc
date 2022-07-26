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


def infer(save_root):
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']
    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_time = AverageMeter()
    all_time = AverageMeter()

    transform = transforms.SemanticKittiProcessData(args.data_process, args.num_points, args.allow_less_points)
    val_dataset = datasets.SemanticKitti(train=False, test=True, transform=transform, num_points=args.num_points, data_root=args.data_root)
    collate = None
    if args.data_process['DOWN_SAMPLE_METHOD'] == 'voxel':
        collate = datasets.Collater(args.data_process)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate)

    model = PointConvSceneFlow()

    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s' % pretrain)

    model = model.cuda()

    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        if args.num_points < 0:
            pos1, pos2, norm1, norm2, flow, batch_path, pos1_mask, pos2_mask, norm1_mask, norm2_mask, flow_mask, inverse_indices, pos1_allpoints, pos2_allpoints = data
        else:
            if args.data_process['DOWN_SAMPLE_METHOD'] == 'random':
                pos1, pos2, norm1, norm2, flow, batch_path, inverse_indices, pos1_allpoints, pos2_allpoints = data
            elif args.data_process['DOWN_SAMPLE_METHOD'] == 'voxel':
                pos1, pos2, norm1, norm2, flow, batch_path, pos1_mask, pos2_mask, norm1_mask, norm2_mask, flow_mask, inverse_indices, pos1_allpoints, pos2_allpoints = data

        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()

        model.eval()

        with torch.no_grad():
            start = time.time()
            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
            time_elapse = time.time() - start
            model_time.update(time_elapse)

            print(f'model time: {model_time.avg}')

            full_flow = pred_flows[0].permute(0, 2, 1)
            pos1 = pos1.cpu().numpy()
            pos2 = pos2.cpu().numpy()
            pred_flow = full_flow.cpu().numpy()
            flow_gt = flow.cpu().numpy()

            if args.num_points < 0:
                np.savez(batch_path[0], current_point=pos1, next_point=pos2, pred_flow=pred_flow)
            else:

                for i in range(len(batch_path)):
                    save_path = os.path.join(save_root, batch_path[i].split('/')[-1])  #! 需要改，把flow的结果存在dataset的每个sequence里，方便4dmos读取
                    print(save_path)
                    pos1_i = pos1[i]
                    pos2_i = pos2[i]
                    pred_flow_i = pred_flow[i]
                    flow_gt_i = flow_gt[i]
                    inverse_indices_i = inverse_indices[i]
                    pos1_allpoint_i = pos1_allpoints[i]
                    pos2_allpoint_i = pos2_allpoints[i]

                    if args.data_process['DOWN_SAMPLE_METHOD'] == 'voxel':
                        pos1_mask_i = pos1_mask[i].astype(np.bool)
                        # print(np.unique(pos1_mask_i,return_counts=True))
                        pos2_mask_i = pos2_mask[i].astype(np.bool)
                        flow_mask_i = flow_mask[i].astype(np.bool)

                        pos1_i = pos1_i[pos1_mask_i]
                        pos2_i = pos2_i[pos2_mask_i]
                        pred_flow_i = pred_flow_i[flow_mask_i]  #! 存flow之前记得把flow 的方向改一下
                        flow_gt_i = flow_gt_i[flow_mask_i]

                    upsampling_pred_flow = pred_flow_i[inverse_indices_i]

                    np.savez(save_path, current_point=pos1_allpoint_i, last_point=pos2_allpoint_i, pred_flow=upsampling_pred_flow)

                    # np.savez(save_path,current_point=pos1_i,last_point=pos2_i,pred_flow=pred_flow_i,flow_gt=flow_gt_i)


if __name__ == "__main__":
    # if not os.path.exists("flow_result"):
    #     os.mkdir("flow_result")
    save_root = 'flow_result/01_voxel_choiceone_0.2_sample_PointConv_112_0.0533_seqall'
    infer(save_root)
