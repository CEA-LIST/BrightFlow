import numpy as np
import random
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter


class BasicAugmentor(object):
    def __init__(self, args, min_scale=-0.2, max_scale=0.6, do_flip=True):

        self.args = args
        
        # spatial augmentation params
        self.crop_size = args.crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

        # artefacts params
        self.num_artefacts = (5, 20)
        self.artefact_size_range = (2, 50)
        self.min_bright = 0.5
        self.max_bright = 2
        self.interv_bright = self.max_bright - self.min_bright

        # full scale warping params with FlyingChairs
        if 'chairs' in args.dataset_train.lower():
            self.H, self.W = 384, 512
        elif 'kitti' in args.dataset_train.lower():
            self.H, self.W = 376, 1242
        elif 'sintel' in args.dataset_train.lower():
            self.H, self.W = 436, 1024
        else:
            raise NotImplementedError

        self.h_max = np.round(self.H * 2**self.max_stretch * 2**self.max_scale).astype(int)
        self.w_max = np.round(self.W * 2**self.max_stretch * 2**self.max_scale).astype(int)


    def global_color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1_aug = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2_aug = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1_aug, img2_aug = np.split(image_stack, 2, axis=0)

        return img1_aug, img2_aug

    def raft_eraser_bidirectional(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = img1.shape[:2]

        def eraser(img):
            mask_eraser = np.ones((ht, wd, 1))
            if np.random.rand() < self.eraser_aug_prob:
                mean_color = np.mean(img.reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    x0 = np.random.randint(0, wd-dx)
                    y0 = np.random.randint(0, ht-dy)
                    img[y0:y0+dy, x0:x0+dx, :] = mean_color
                    mask_eraser[y0:y0+dy, x0:x0+dx, :] = 0.
            return img.clip(0, 255), mask_eraser

        if np.random.random() > 0.5:
            img1, mask_eraser1 = eraser(img1)
            mask_eraser2 = np.ones((ht, wd, 1))
        else:
            img2, mask_eraser2 = eraser(img2)
            mask_eraser1 = np.ones((ht, wd, 1))
        return img1, img2, mask_eraser1, mask_eraser2


    def spatial_transform(self, img1, img2):
        """ Spatial augmentation """
        # random flip
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]

        # randomly sample scale
        if np.random.rand() < self.spatial_aug_prob:
            ht, wd = img1.shape[:2]
            min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(ht), 
                (self.crop_size[1] + 8) / float(wd))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale
            if np.random.rand() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            
            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        if self.args.use_full_size_warping:
            # When using full-size warping, uncropped image and the position of the crop have to be stored
            uncropped_img1, uncropped_img2 = img1.copy(), img2.copy()
            H, W, _ = uncropped_img1.shape
            orig_dims = np.array([H, W])
            
            if img1.shape[0] == self.crop_size[0]:
                y0 = 0
            else:
                y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])

            if img1.shape[1] == self.crop_size[1]:
                x0 = 0
            else:
                x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

            pad_right = uncropped_img1.shape[1] - (x0 + self.crop_size[1])
            pad_bottom = uncropped_img1.shape[0] - (y0 + self.crop_size[0])
            pad_params = np.array([x0, pad_right, y0, pad_bottom])

            # if self.args.batch_size == 1:
            uncropped_img1 = np.pad(uncropped_img1, ((0, self.h_max - H), (0, self.w_max - W), (0, 0)))
            uncropped_img2 = np.pad(uncropped_img2, ((0, self.h_max - H), (0, self.w_max - W), (0, 0)))

            return img1, img2, uncropped_img1, uncropped_img2, pad_params, orig_dims
            
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        
        return img1, img2

    def selfsup_transform(self, x):
        num_channels = x.shape[-1]
        x = x[64:-64, 64:-64]
        x_resized = cv2.resize(x, (self.args.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR)
        if num_channels == 1:
            return np.expand_dims(np.floor(x_resized), axis=-1)
        return x_resized


class Augmentor(BasicAugmentor):
    def __init__(self, args):
        super(Augmentor, self).__init__(args)

    def __call__(self, img1, img2):
        example = {}

        if self.args.use_full_size_warping:
            img1, img2, uncropped_img1, uncropped_img2, pad_params, orig_dims = self.spatial_transform(img1, img2)

            example.update({
                'ims_uncropped': np.stack([np.ascontiguousarray(uncropped_img1), np.ascontiguousarray(uncropped_img2)]),
                'pad_params': pad_params,
                'orig_dims': orig_dims,
            })

        else:
            img1, img2 = self.spatial_transform(img1, img2)
        
        if self.args.no_photo_aug:
            img1_aug, img2_aug = img1, img2
            ht, wd = img1_aug.shape[:2]
            mask_eraser1 = np.ones((ht, wd, 1))
            mask_eraser2 = np.ones_like(mask_eraser1)
        else:
            img1_aug, img2_aug = self.global_color_transform(img1, img2)
            
            if self.args.random_eraser:
                img1_aug, img2_aug, mask_eraser1, mask_eraser2 = self.raft_eraser_bidirectional(img1_aug, img2_aug)
            else:
                ht, wd = img1_aug.shape[:2]
                mask_eraser1 = np.ones((ht, wd, 1))
                mask_eraser2 = np.ones_like(mask_eraser1)

        img1_aug_stud = self.selfsup_transform(img1_aug)
        img2_aug_stud = self.selfsup_transform(img2_aug)
        mask_eraser1_stud = self.selfsup_transform(mask_eraser1)
        mask_eraser2_stud = self.selfsup_transform(mask_eraser2)
        valid = np.ones((1, 1, *self.crop_size))

        example.update({
            'ims': np.stack([np.ascontiguousarray(img1), np.ascontiguousarray(img2)]),
            'ims_aug': np.stack([np.ascontiguousarray(img1_aug), np.ascontiguousarray(img2_aug)]),
            'ims_aug_stud': np.stack([np.ascontiguousarray(img1_aug_stud), np.ascontiguousarray(img2_aug_stud)]),
            'masks_eraser': np.stack([np.ascontiguousarray(mask_eraser1), np.ascontiguousarray(mask_eraser2)]),
            'masks_eraser_stud': np.stack([np.ascontiguousarray(mask_eraser1_stud), np.ascontiguousarray(mask_eraser2_stud)]),
            'valid': np.ascontiguousarray(valid),
        })

        return example
