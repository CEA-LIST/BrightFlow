
import numpy as np
import random
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from .augmentations import BasicAugmentor

class AugmentorCorrections(BasicAugmentor):
    def __init__(self, args):
        super(AugmentorCorrections, self).__init__(args)

    def spatial_transform_correc_full_size_warping(self, img1, img2, img1_correc, img2_correc):

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
            img1_correc = cv2.resize(img1_correc, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2_correc = cv2.resize(img2_correc, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                img1_correc = img1_correc[:, ::-1]
                img2_correc = img2_correc[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                img1_correc = img1_correc[::-1, :]
                img2_correc = img2_correc[::-1, :]

        uncropped_img1, uncropped_img2 = img1.copy(), img2.copy()
        uncropped_img1_correc, uncropped_img2_correc = img1_correc.copy(), img2_correc.copy()

        H, W, _ = uncropped_img1.shape
        orig_dims = np.array([H, W])
        
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img1_correc = img1_correc[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2_correc = img2_correc[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        pad_right = uncropped_img1.shape[1] - (x0 + self.crop_size[1])
        pad_bottom = uncropped_img1.shape[0] - (y0 + self.crop_size[0])
        pad_params = np.array([x0, pad_right, y0, pad_bottom])

        uncropped_img1 = np.pad(uncropped_img1, ((0, self.h_max - H), (0, self.w_max - W), (0, 0)))
        uncropped_img2 = np.pad(uncropped_img2, ((0, self.h_max - H), (0, self.w_max - W), (0, 0)))
        uncropped_img1_correc = np.pad(uncropped_img1_correc, ((0, self.h_max - H), (0, self.w_max - W), (0, 0)))
        uncropped_img2_correc = np.pad(uncropped_img2_correc, ((0, self.h_max - H), (0, self.w_max - W), (0, 0)))

        return img1, img2, uncropped_img1, uncropped_img2, img1_correc, img2_correc, uncropped_img1_correc, uncropped_img2_correc, pad_params, orig_dims

    def __call__(self, img1, img2):
        example = {}
        if self.args.use_full_size_warping:

            if self.args.no_photo_aug:
                img1_aug, img2_aug = img1, img2
            else:
                img1_aug, img2_aug = self.global_color_transform(img1, img2)

            img1, img2, uncropped_img1, uncropped_img2, img1_aug, img2_aug, uncropped_img1_aug, uncropped_img2_aug, pad_params, orig_dims \
                = self.spatial_transform_correc_full_size_warping(img1, img2, img1_aug, img2_aug)
            
        else:
            img1, img2 = self.spatial_transform(img1, img2)
            if self.args.no_photo_aug:
                img1_aug, img2_aug = img1, img2
            else:
                img1_aug, img2_aug = self.global_color_transform(img1, img2)

        if self.args.no_photo_aug:
            ht, wd = img1_aug.shape[:2]
            mask_eraser1 = np.ones((ht, wd, 1))
            mask_eraser2 = np.ones_like(mask_eraser1)
        else:
            if self.args.random_eraser:
                img1_aug, img2_aug, mask_eraser1, mask_eraser2 = self.raft_eraser_bidirectional(img1_aug, img2_aug)
            else:
                ht, wd = img1_aug.shape[:2]
                mask_eraser1 = np.ones((ht, wd, 1))
                mask_eraser2 = np.ones_like(mask_eraser1)

        if self.args.use_full_size_warping:
            x0, _, y0, _ = pad_params
            uncropped_img1_aug[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] = img1_aug
            uncropped_img2_aug[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] = img2_aug
            
            example.update({
                'ims_uncropped': np.stack([np.ascontiguousarray(uncropped_img1), np.ascontiguousarray(uncropped_img2)]),
                'ims_aug_uncropped': np.stack([np.ascontiguousarray(uncropped_img1_aug), np.ascontiguousarray(uncropped_img2_aug)]),
                'pad_params': pad_params,
                'orig_dims': orig_dims,
            })

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
