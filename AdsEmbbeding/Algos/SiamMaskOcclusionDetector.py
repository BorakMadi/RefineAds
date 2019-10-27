import argparse
from os.path import isfile

import numpy as np
import cv2
import torch

from Interface.OcclusionDetector import OcclusionDetector
from .libs.SiamMask.utils.config_helper import load_config
from .libs.SiamMask.utils.load_helper import load_pretrain
from .libs.SiamMask.tools.test import *

from skimage.transform import warp
from skimage.transform import ProjectiveTransform

SHOW_RESULTS = True

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='Algos/libs/SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth',
                    type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='Algos/libs/SiamMask/experiments/siammask_sharp/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


def normalize(points):
    for row in points:
        row /= points[-1]
    return points


def project(fp, H):
    # transform fp
    fp = np.asarray([fp[0], fp[1], 1])
    fp_transformed = np.dot(H, fp)
    # normalize hom. coordinates
    fp_transformed = normalize(fp_transformed)

    return np.asarray([fp_transformed[0], fp_transformed[1]])



class SiamMaskOcclusionDetector(OcclusionDetector):

    def __init__(self, mask_size=(500,500)):

        super().__init__()
        self.mask_size = mask_size
        self.curr_index = 0
        self.state = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        self.cfg = load_config(args)
        from .libs.SiamMask.experiments.siammask_sharp.custom import Custom
        self.siammask = Custom(anchors=self.cfg['anchors'])

        self.curr_mask = np.ones(mask_size)

        if args.resume:
            assert isfile(args.resume), '{} does not exist.'.format(args.resume)
            self.siammask = load_pretrain(self.siammask, args.resume)

        self.siammask.eval().to(self.device)

        # === update_mask ===
    def update_mask(self, ad_region):

        H = ad_region.get_homography(self.mask_size)

        im = ad_region.original_frame

        if self.curr_index == 0:  # init
            margin = 40
            x, y, w, h = cv2.boundingRect(np.asarray(ad_region.get_corners(), dtype=np.float32))
            x -= margin
            y -= margin
            w += margin
            h += margin
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            self.state = siamese_init(im, target_pos, target_sz, self.siammask, self.cfg['hp'], device=self.device)  # init tracker
        elif self.curr_index > 0:  # tracking
            state = siamese_track(self.state, im, mask_enable=True, refine_enable=True, device=self.device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            trans_mask = warp(mask, H, output_shape=self.mask_size)

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

            if SHOW_RESULTS:
                for c in ad_region.get_corners():
                    cv2.circle(im, (c[0][0], c[0][1]), 5, (0,255,255), -1)
                cv2.imshow('SiamMask', im)
                cv2.imshow('Mask', 255*mask.astype(np.uint8))
                cv2.imshow('trans_mask', trans_mask)
                cv2.waitKey(1)


            self.curr_mask = trans_mask

        self.curr_index += 1

        ad_region.update_mask(self.curr_mask)

        return ad_region
