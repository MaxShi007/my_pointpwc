import numpy as np
# from torchsparse.utils.quantize import sparse_quantize
from itertools import repeat
from typing import List, Tuple, Union
from icecream import ic
import time
import glob

# pc1=np.load('/root/sgb_repo/PointPWC/SAVE_PATH/KITTI_processed_occ_final/000000/pc1.npy')
# pc2=np.load('/root/sgb_repo/PointPWC/SAVE_PATH/KITTI_processed_occ_final/000000/pc2.npy')
# print(pc1.shape,pc2.shape)
def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x -= np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = coords / voxel_size # coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    outputs = [coords/10]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs


data_root='/share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point'

paths=glob.glob(data_root+'/*.npz')
for path in paths:

    data=np.load(path)
    current=data['current_point']
    ic(current.shape,current)
    # start=time.time()
    min=np.min(current,axis=0,keepdims=True)
    current-=min
    ic(current.shape,current)
    coords,indices=sparse_quantize(current,voxel_size=0.1,return_index=True)
    coords=coords.astype('float32')
    # print(time.time()-start)
    
    ic(coords,current[indices])

    break


# input_size=5
# voxel_size=0.1
# inputs = np.random.uniform(-100, 100, size=(input_size, 4))
# labels = np.random.choice(10, size=input_size)
# print(inputs)
# coords, feats = inputs[:, :3], inputs
# ic(coords)
# coords -= np.min(coords, axis=0, keepdims=True)
# ic(coords,type(coords[0,0]))
# coords, indices = sparse_quantize(coords,voxel_size,return_index=True)
# ic(coords,indices)