import numpy as np
from torchsparse.utils.quantize import sparse_quantize

pc1=np.load('/root/sgb_repo/PointPWC/SAVE_PATH/KITTI_processed_occ_final/000000/pc1.npy')
pc2=np.load('/root/sgb_repo/PointPWC/SAVE_PATH/KITTI_processed_occ_final/000000/pc2.npy')
print(pc1.shape,pc2.shape)