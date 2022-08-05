import numpy as np
from torch.utils.data import Dataset
import glob
import os
import torch

import sys

sys.path.append("/root/sgb_repo/PointPWC")
import transforms
# from transforms.transforms import SemanticKittiProcessData

# __all__=['SemanticKitti']


class SemanticKitti(Dataset):
    '''
    args:
        use_all: if True, use all data including trainset validset testset, otherwise, use only the training(00-09,10) data
    '''

    def __init__(self, train, transform, num_points, data_root, use_all=False, test=False):
        self.root = data_root
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.test = test

        self.samples = self.make_dataset(use_all=use_all)
        # print(len(self.samples))
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in: " + self.root + "\n"))

    def __len__(self):
        return (len(self.samples))

    def __getitem__(self, index):
        current_point, last_point, flow_gt = self.pc_loader(self.samples[index])
        current_point_transformed, last_point_transformed, flow_gt_transformed, inverse_current, allpoints_current, allpoints_last = self.transform([current_point, last_point, flow_gt])

        current_point_norm = current_point_transformed
        last_point_norm = last_point_transformed

        return current_point_transformed, last_point_transformed, current_point_norm, last_point_norm, flow_gt_transformed, self.samples[index], inverse_current, allpoints_current, allpoints_last

    def pc_loader(self, path):
        data = np.load(path)
        current_point = data['current_point'].astype('float32')
        last_point = data['last_point'].astype('float32')
        flow_gt = data['flow_gt'].astype('float32')

        assert (len(current_point) == len(flow_gt))
        # print(len(current_point))
        return current_point, last_point, flow_gt

    def make_dataset(self, use_all):
        paths = []
        train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41']  # ,'30', '31', '32', '33', '34', '40'
        # train_sequences = ['04']
        val_sequences = ['08']  # ,'35', '36', '37', '38', '39', '41'

        if self.test:
            print(f"test: {self.test}")
            test_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41']
            print(f"test_sequences: {test_sequences}")
            paths = [os.path.join(root, file) for root, dir, files in os.walk(self.root) for file in files if file.split('_')[0] in test_sequences]
        else:
            if self.train:
                if use_all:
                    paths = sorted(glob.glob(os.path.join(self.root, '*.npz')))
                else:  # use only the training data include kittiroad
                    for root, dir, files in os.walk(self.root):
                        files = sorted(files)
                        for file in files:
                            sequence = file.split('_')[0]
                            if sequence in train_sequences:
                                paths.append(os.path.join(root, file))
            else:  # validset incklude kittiroad
                paths = [os.path.join(root, file) for root, dir, files in os.walk(self.root) for file in files if file.split('_')[0] in val_sequences]
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


class Collater():

    def __init__(self, data_process_args):
        self.pad_method = data_process_args['PAD_METHOD']

    def __call__(self, batch):
        pos1, pos2, norm1, norm2, flow, path, inverse_current, allpoints_current, allpoints_last = zip(*batch)
        batch_len = len(pos1)

        max_pos1_len = max([len(x) for x in pos1])
        max_pos2_len = max([len(x) for x in pos2])
        max_norm1_len = max([len(x) for x in norm1])
        max_norm2_len = max([len(x) for x in norm2])
        max_flow_len = max([len(x) for x in flow])

        pos1_pad = np.zeros((batch_len, max_pos1_len, 3))
        pos2_pad = np.zeros((batch_len, max_pos2_len, 3))
        norm1_pad = np.zeros((batch_len, max_norm1_len, 3))
        norm2_pad = np.zeros((batch_len, max_norm2_len, 3))
        flow_pad = np.zeros((batch_len, max_flow_len, 3))

        pos1_mask = np.zeros((batch_len, max_pos1_len))
        pos2_mask = np.zeros((batch_len, max_pos2_len))
        norm1_mask = np.zeros((batch_len, max_norm1_len))
        norm2_mask = np.zeros((batch_len, max_norm2_len))
        flow_mask = np.zeros((batch_len, max_flow_len))

        for i in range(batch_len):
            pos1_pad[i, :len(pos1[i]), :] = pos1[i]
            pos2_pad[i, :len(pos2[i]), :] = pos2[i]
            norm1_pad[i, :len(norm1[i]), :] = norm1[i]
            norm2_pad[i, :len(norm2[i]), :] = norm2[i]
            flow_pad[i, :len(flow[i]), :] = flow[i]

            pos1_mask[i, :len(pos1[i])] = 1
            pos2_mask[i, :len(pos2[i])] = 1
            norm1_mask[i, :len(norm1[i])] = 1
            norm2_mask[i, :len(norm2[i])] = 1
            flow_mask[i, :len(flow[i])] = 1

            # sample padding
            if self.pad_method == 'sample':
                diff_count_pos1 = max_pos1_len - len(pos1[i])
                diff_count_pos2 = max_pos2_len - len(pos2[i])

                diff_pos1 = np.random.choice(len(pos1[i]), diff_count_pos1, replace=True, p=None)
                diff_pos2 = np.random.choice(len(pos2[i]), diff_count_pos2, replace=True, p=None)

                pos1_pad[i, len(pos1[i]):, :] = pos1[i][diff_pos1]
                pos2_pad[i, len(pos2[i]):, :] = pos2[i][diff_pos2]
                norm1_pad[i, len(norm1[i]):, :] = norm1[i][diff_pos1]
                norm2_pad[i, len(norm2[i]):, :] = norm2[i][diff_pos2]
                flow_pad[i, len(flow[i]):, :] = flow[i][diff_pos1]

        pos1 = torch.as_tensor(pos1_pad.astype('float32'))
        pos2 = torch.as_tensor(pos2_pad.astype('float32'))
        norm1 = torch.as_tensor(norm1_pad.astype('float32'))
        norm2 = torch.as_tensor(norm2_pad.astype('float32'))
        flow = torch.as_tensor(flow_pad.astype('float32'))
        # path=torch.as_tensor(path)
        return (pos1, pos2, norm1, norm2, flow, path, pos1_mask, pos2_mask, norm1_mask, norm2_mask, flow_mask, inverse_current, allpoints_current, allpoints_last)


def test_collate(batch):
    pos1, pos2, norm1, norm2, flow, path = zip(*batch)

    pos1 = torch.as_tensor(pos1)
    pos2 = torch.as_tensor(pos2)
    norm1 = torch.as_tensor(norm1)
    norm2 = torch.as_tensor(norm2)
    flow = torch.as_tensor(flow)

    return (pos1, pos2, norm1, norm2, flow, path)


if __name__ == '__main__':
    dataset = SemanticKitti(train=True, transform=transforms.SemanticKittiProcessData('random', 8192), num_points=1024, data_root='/share/sgb/semantic_kitti/SemanticKitti_Flow_Dataset_1_non_ground_point', use_all=True)
    # dataset.getitem(0)
