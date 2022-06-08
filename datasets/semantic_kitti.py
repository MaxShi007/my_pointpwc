import numpy as np
from torch.utils.data import Dataset
import glob
import os

import sys 
sys.path.append("/root/sgb_repo/PointPWC") 
import transforms
# from transforms.transforms import SemanticKittiProcessData

# __all__=['SemanticKitti']

class SemanticKitti(Dataset):
    '''
    args:
        use_all: if True, use all the data, otherwise, use only the training(00-09,10) data
    '''
    def __init__(self,train,transform,num_points,data_root,use_all=True):
        self.root=data_root
        self.train=train
        self.transform=transform
        self.num_points=num_points

        self.samples=self.make_dataset(use_all=use_all)
        # print(len(self.samples))
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in: " + self.root + "\n"))

    def __len__(self):
        return(len(self.samples))

    def __getitem__(self, index):
        current_point,last_point,flow_gt=self.pc_loader(self.samples[index])
        current_point_transformed,last_point_transformed,flow_gt_transformed=self.transform([current_point,last_point,flow_gt])

        current_point_norm=current_point_transformed
        last_point_norm=last_point_transformed

        return current_point_transformed,last_point_transformed,current_point_norm,last_point_norm,flow_gt_transformed,self.samples[index]

    def pc_loader(self,path):
        data=np.load(path)
        current_point=data['current_point'].astype('float32')
        last_point=data['last_point'].astype('float32')
        flow_gt=data['flow_gt'].astype('float32')

        assert(len(current_point)==len(flow_gt))
        # print(len(current_point))
        return current_point,last_point,flow_gt
        

    def make_dataset(self,use_all):
        paths=[]
        if self.train:
            if use_all:
                paths=sorted(glob.glob(os.path.join(self.root,'*.npz')))
            else:
                for root,dir,files in os.walk(self.root):
                    files=sorted(files)
                    for file in files:
                        sequence=file.split('_')[0]
                        if sequence in ['00','01','02','03','04','05','06','07','09','10']:
                            paths.append(os.path.join(root,file))
        else:
            paths=[os.path.join(root,file) for root,dir,files in os.walk(self.root) for file in files if file.split('_')[0] in ['08']]
        return paths

    def __repr__(self) -> str:
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
                    

            


if __name__=='__main__':
    dataset=SemanticKitti(train=True,transform=transforms.SemanticKittiProcessData('random',8192),num_points=1024,data_root='/share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point',use_all=True)
    dataset.getitem(0)
    