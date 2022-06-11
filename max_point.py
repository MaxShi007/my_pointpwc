import numpy as np
import open3d as o3d
import glob
from  multiprocessing import Process,Pool
from tqdm import tqdm


def down_sample(points,voxel_size):
    '''
    returns:
        pc_ds_points: ndarray
        pc_ds_index: [[]]
    '''
    pc=o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(points)
    pc_min_bound = pc.get_min_bound()
    pc_max_bound = pc.get_max_bound()
    pc_ds=pc.voxel_down_sample_and_trace(voxel_size=voxel_size,min_bound=pc_min_bound,max_bound=pc_max_bound)
    pc_ds_points=np.asarray(pc_ds[0].points)
    pc_ds_index=[list(index) for index in pc_ds[2]]
    
    return pc_ds_points,pc_ds_index

def get_max_point(data_root,voxel_size):
    max_point_len=0
    max_point_path=''

    paths=glob.glob(data_root+'/*.npz')
    for path in tqdm(paths):
        data=np.load(path)
        current_point=data['current_point']
        last_point=data['last_point']
        flow_gt=data['flow_gt']

        current_point_ds,index=down_sample(current_point,voxel_size)

        if len(current_point_ds)>max_point_len:
            max_point_len=len(current_point_ds)
            max_point_path=path+'_current'
        

    return max_point_len,max_point_path
if __name__=='__main__':
    
    data_root='/share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point'
    

    voxel_size=0.3
    max_point,path=get_max_point(data_root,voxel_size) # 10469 /share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point/00_002357.npz_current
    print(max_point,path)

    voxel_size=0.2
    max_point,path=get_max_point(data_root,voxel_size) # 16958 /share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point/00_002357.npz_current
    print(max_point,path)

    voxel_size=0.1
    max_point,path=get_max_point(data_root,voxel_size) # 32282 /share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point/00_002357.npz_current
    print(max_point,path)



        
        


