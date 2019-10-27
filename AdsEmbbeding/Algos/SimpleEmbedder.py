from Interface.Embedder import Embedder
import cv2
import numpy as np
from skimage.transform import warp
from skimage.transform import ProjectiveTransform

class SimpleEmbedder(Embedder):

    def embed(self, ad_region):
        ad = ad_region.get_embedding_image()
        frame = ad_region.edited_frame

        _mask = ad_region.get_mask((ad.shape[1], ad.shape[0]))
        # cv2.imshow("O mask",_mask)

        mask = np.ones((_mask.shape[0], _mask.shape[1], 3))
        mask[:, :, 0] = _mask
        mask[:, :, 1] = _mask
        mask[:, :, 2] = _mask

        H = ad_region.get_homography((ad.shape[0], ad.shape[1]))
        trans_img = warp(ad * mask, ProjectiveTransform(matrix=H).inverse, output_shape=(frame.shape[0], frame.shape[1]))

        change_indices = trans_img != 0
        frame[change_indices] = trans_img[change_indices]

        return ad_region

