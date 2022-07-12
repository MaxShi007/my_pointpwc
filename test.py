from cgi import print_arguments
import numpy as np
from torchsparse.utils.quantize import sparse_quantize
from itertools import repeat
from typing import List, Tuple, Union
from icecream import ic
import time
import glob
from tqdm import tqdm


# data_root='/share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point'

# paths=glob.glob(data_root+'/*.npz')
# for path in paths:
#     print(path)

#     start=time.time()
#     data=np.load(path)
#     current=data['current_point']
#     last=data['last_point']
#     flow_gt=data['flow_gt']

#     orig_current=current.copy()
#     orig_last=last.copy()

#     min_current=np.min(current,axis=0,keepdims=True)
#     min_last=np.min(last,axis=0,keepdims=True)

#     current-=min_current
#     last-=min_last
#     _,indices_current=sparse_quantize(current,voxel_size=0.2,return_index=True)
#     _,indices_last=sparse_quantize(last,voxel_size=0.2,return_index=True)

#     current_point_ds=orig_current[indices_current]
#     last_point_ds=orig_last[indices_last]
#     flow_gt_ds=flow_gt[indices_current]
#     print(time.time()-start)

#     np.savez("./ds_point.npz",current_point=orig_current,last_point=orig_last,flow_gt=flow_gt,current_point_ds=current_point_ds,last_point_ds=last_point_ds,flow_gt_ds=flow_gt_ds)
#     break
    





    

def get_max_point(data_root,voxel_size):
    max_point_len=0
    max_point_path=''

    paths=glob.glob(data_root+'/*.npz')
    for path in tqdm(paths):
        data=np.load(path)
        current_point=data['current_point']
        last_point=data['last_point']
        flow_gt=data['flow_gt']

        current_point_ds,index=sparse_quantize(current_point,voxel_size=voxel_size,return_index=True)

        if len(current_point_ds)>max_point_len:
            max_point_len=len(current_point_ds)
            max_point_path=path+'_current'
            # print(max_point_len)
        
    return max_point_len,max_point_path


def test2(data_root,voxel_size):
    paths=glob.glob(data_root+'/*.npz')
    for path in tqdm(paths):
        print(path)
        data=np.load(path)
        current_point=data['current_point']
        last_point=data['last_point']
        flow_gt=data['flow_gt']

        current_point-=np.min(current_point, axis=0, keepdims=True)
        coords_all=np.floor(current_point / voxel_size).astype(np.int32) 
        coords_ds,index=sparse_quantize(current_point,voxel_size=voxel_size,return_index=True)
        # print(coords_all.shape,coords_ds)
        # print(coords_all.min(),coords_all.max())
        # ic(coords_ds.shape,coords_ds)
        # ic(np.min(coords_ds,axis=0),np.max(coords_ds,axis=0))
        ic(coords_ds,index)
        break

def test3(data_root,voxel_size):
    paths=glob.glob(data_root+'/*.npz')
    for path in tqdm(paths):
        print(path)
        data=np.load(path)
        current_point=data['current_point']
        last_point=data['last_point']
        flow_gt=data['flow_gt']

        current_point-=np.min(current_point, axis=0, keepdims=True)

        coords_ds,index1,inverse=sparse_quantize(current_point,voxel_size=voxel_size,return_index=True,return_inverse=True)
        ic(coords_ds,index1,inverse)
        coords_ds,index2,inverse=sparse_quantize(current_point,voxel_size=voxel_size,return_index=True,return_inverse=True)
        ic(coords_ds,index2,inverse)
        if (index1==index2).all():
            print('same')
        index,count=np.unique(inverse,return_counts=True)
        ic(count.max(),count.min())
        break

def test4(voxel_size):
    a=np.array([[1,1,1],[2,3,2],[3,3,3],[1.21,1.05,1.05],[1.05,1.05,1.05]])
    coords = np.floor(a / voxel_size).astype(np.int32)
    print(coords)



if __name__=='__main__':
    
    data_root='/share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point'
    
#################################################
    # voxel_size=0.3
    # max_point,path=get_max_point(data_root,voxel_size) 
    # print(max_point,path)

    # voxel_size=0.2
    # max_point,path=get_max_point(data_root,voxel_size)
    # print(max_point,path)

    # voxel_size=0.1
    # max_point,path=get_max_point(data_root,voxel_size) 
    # print(max_point,path)
####################################################
#     voxel_size=0.2
#     test2(data_root,voxel_size)
# ##################################################
#     voxel_size=0.2
#     test3(data_root,voxel_size)
#############################################
    voxel_size=0.2
    test4(voxel_size)