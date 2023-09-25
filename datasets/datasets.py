import os
import os.path as osp
from glob import glob
import numpy as np
import torch
import torch.utils.data as data
from skimage.color import rgb2gray

from datasets.utils_data import read_gen, readFlowKITTI


class FlowDataset(data.Dataset):
    def __init__(self, args, augmentor=None, sparse=False, is_training=True):
        self.args = args
        if augmentor is not None:
            self.augmentor = augmentor(args)
        else:
            self.augmentor = None
        self.sparse = sparse
        self.is_training = is_training

        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.pseudo_gt_list = []

    def __getitem__(self, index):

        index = index % len(self.image_list)

        im1 = read_gen(self.image_list[index][0])
        im2 = read_gen(self.image_list[index][1])
        im1 = np.array(im1)
        im2 = np.array(im2)

        # grayscale images
        if len(im1.shape) == 2:
            im1 = np.tile(im1.astype(np.uint8)[...,None], (1, 1, 3))
            im2 = np.tile(im2.astype(np.uint8)[...,None], (1, 1, 3))

        elif self.args.data_in_grayscale:
            im1 = np.tile(rgb2gray(im1)[...,None] * 255, (1, 1, 3)).astype(np.uint8)
            im2 = np.tile(rgb2gray(im2)[...,None] * 255, (1, 1, 3)).astype(np.uint8)

        else:
            im1 = im1[..., :3].astype(np.uint8)
            im2 = im2[..., :3].astype(np.uint8)
        # else:
        if self.augmentor is not None:
            example = self.augmentor(im1, im2)

        else:
            example = {
                'ims': np.stack([im1, im2])
                }

        for key, value in example.items():
            if key in ['pad_params', 'orig_dims', 'offsets', 'valid', 'pseudo_gt']:
                example[key] = torch.from_numpy(value)
            else:
                example[key] = torch.from_numpy(value).permute(0, 3, 1, 2).float()

        example['index'] = [index, self.extra_info[index]]

        if not self.is_training:

            if self.sparse:
                flow_occ, valid_occ = readFlowKITTI(self.flow_occ_list[index])
                flow_noc, valid_noc = readFlowKITTI(self.flow_noc_list[index])

                flow_occ = np.array(flow_occ).astype(np.float32)
                flow_noc = np.array(flow_noc).astype(np.float32)
                valid_occ = np.expand_dims(valid_occ, axis=0)
                valid_noc = np.expand_dims(valid_noc, axis=0)
                
                example['flow_occ'] = torch.from_numpy(flow_occ).permute(2, 0, 1).float()
                example['valid_occ'] = torch.from_numpy(valid_occ)
                example['flow_noc'] = torch.from_numpy(flow_noc).permute(2, 0, 1).float()
                example['valid_noc'] = torch.from_numpy(valid_noc)

            else:

                flow = read_gen(self.flow_list[index])
                flow = np.array(flow).astype(np.float32)
                flow = torch.from_numpy(flow).permute(2, 0, 1).float()
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000).unsqueeze(0)

                example['flow'] = flow
                example['valid'] = valid

        return example

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, args, augmentor, is_training=True, split='training', root='/path/to/sintel/', dstype='final'):
        super(MpiSintel, self).__init__(args, augmentor=augmentor, sparse=False, is_training=is_training)

        if split == 'training':
            image_root = osp.join(root, 'test', dstype)

            for scene in os.listdir(image_root):
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list)-1):
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    frame1_id = image_list[i].split('/')[-1].split('.')[0]
                    frame2_id = image_list[i+1].split('/')[-1].split('.')[0]
                    self.extra_info += [[f'{scene}_{frame1_id}', f'{scene}_{frame2_id}']]

        elif split == 'validation':
            image_root = osp.join(root, 'training', dstype)
            flow_root = osp.join(root, 'training', 'flow')

            for scene in os.listdir(image_root):
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list)-1):
                    self.image_list += [ [image_list[i], image_list[i+1]] ]

                    self.extra_info += [ (scene, i) ] # scene and frame_id

                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class KITTI(FlowDataset):
    def __init__(self, args, augmentor, is_training=True, split='training', root='/path/to/kitti_data/data_scene_flow/'):
        super(KITTI, self).__init__(args, augmentor=augmentor, sparse=True, is_training=is_training)

        images1 = []
        images2 = []
        self.flow_occ_list = []
        self.flow_noc_list = []

        if split == 'training':
            root = osp.join(root, 'testing')

            for dir in ['image_2', 'image_3']:
                for i in range(200):
                    images1 += sorted(glob(osp.join(root, f'{dir}/{str(i).zfill(6)}_*.png')))[:-1]
                    images2 += sorted(glob(osp.join(root, f'{dir}/{str(i).zfill(6)}_*.png')))[1:]

        elif split == 'validation':
            root = osp.join(root, 'training')

            for i in range(200):
                images1 += [osp.join(root, f'image_2/{str(i).zfill(6)}_10.png')]
                images2 += [osp.join(root, f'image_2/{str(i).zfill(6)}_11.png')]
            for i in range(200):

                self.flow_occ_list += [osp.join(root, f'flow_occ/{str(i).zfill(6)}_10.png')]
                self.flow_noc_list += [osp.join(root, f'flow_noc/{str(i).zfill(6)}_10.png')]
            
        for im1, im2 in zip(images1, images2):
            dir = im1.split('/')[-2]
            frame1_id = im1.split('/')[-1].split('.')[0]
            frame2_id = im2.split('/')[-1].split('.')[0]
            self.extra_info += [ [f'{dir}_{frame1_id}', f'{dir}_{frame2_id}'] ]
            self.image_list += [ [im1, im2] ]


class HD1K(FlowDataset):
    def __init__(self, args, augmentor, is_training=True, split='training', root='/path/to/HD1K'):
        super(HD1K, self).__init__(args, augmentor=augmentor, sparse=True, is_training=is_training)

        self.flow_occ_list = []
        self.flow_noc_list = []

        if split == 'validation':

            seq_ix = 0
            while True:
                flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
                images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

                if len(flows) == 0:
                    break

                for i in range(len(flows)-1):
                    self.flow_occ_list += [flows[i]]
                    self.flow_noc_list += [flows[i]]
                    self.image_list += [ [images[i], images[i+1]] ]
                    frame1_id = images[i].split('/')[-1].split('.')[0]
                    frame2_id = images[i+1].split('/')[-1].split('.')[0]
                    self.extra_info += [ [frame1_id, frame2_id] ]

                seq_ix += 1

        elif split == 'training':

            seq_ix = 0
            while True:
                images = sorted(glob(os.path.join(root, 'hd1k_challenge', 'image_2/%06d_*.png' % seq_ix)))

                if len(flows) == 0:
                    break

                for i in range(len(flows)-1):
                    self.image_list += [ [images[i], images[i+1]] ]
                    frame1_id = images[i].split('/')[-1].split('.')[0]
                    frame2_id = images[i+1].split('/')[-1].split('.')[0]
                    self.extra_info += [ [frame1_id, frame2_id] ]

                seq_ix += 1


class Chairs(FlowDataset):
    def __init__(self, args, augmentor, is_training=True, split='training', root='/path/to/FlyingChairs_release/data'):
        super(Chairs, self).__init__(args, augmentor=augmentor, sparse=False, is_training=is_training)

        self.args = args

        flows = sorted(glob(osp.join(root, '*.flo')))
        split_list = np.loadtxt('/path/to/FlyingChairs_release/FlyingChairs_train_val.txt', dtype=np.int32)

        images = sorted(glob(osp.join(root, '*.ppm')))
        assert (len(images)//2 == len(flows))

        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]
                frame1_id = images[2*i].split('/')[-1].split('.')[0]
                frame2_id = images[2*i+1].split('/')[-1].split('.')[0]
                self.extra_info += [ [frame1_id, frame2_id] ]
