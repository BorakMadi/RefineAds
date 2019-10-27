import numpy as np
import cv2


class AdRegion(object):

    def __init__(self, corners, original_frame, edited_frame, image_to_embed, k=1):
        self.region_corners = corners
        self.original_frame = original_frame
        self.edited_frame = edited_frame
        self.embedding_image = image_to_embed
        self.mask = np.ones((100, 100))
        self.tracking_updated = True
        self.mask_updated = True
        self.curr_frame_index = -1

        self.k = k
        self.next_prev_regions = {}

    def get_homography(self, shape):
        h, w = shape

        dst_pts = np.array(self.region_corners, dtype="float32")

        offset = 0

        src_pts = np.array([[w + offset, h + offset],
                            [0 - offset, h + offset],
                            [0 - offset, 0 - offset],
                            [w + offset, 0 - offset]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        return M

    def get_mask(self, shape):
        mask = cv2.resize(self.mask, shape, interpolation=cv2.INTER_CUBIC)

        return mask

    def get_embedding_image(self):
        return self.embedding_image

    def get_corners(self):
        return self.region_corners

    def update_corners(self, corners):
        self.region_corners = corners.copy()
        self.tracking_updated = True

    def update_mask(self, mask):
        self.mask = mask
        self.mask_updated = True


    def update_frame(self, original_frame, edited_frame):
        self.original_frame = original_frame
        self.edited_frame = edited_frame

        self.tracking_updated = False
        self.mask_updated = False
        self.curr_frame_index += 1

        # prev_region = AdRegion(self.region_corners, self.original_frame, self.edited_frame, self.embedding_image, k=self.k)
        # prev_region.mask = self.mask
        # prev_region.mask_updated = True
        # prev_region.tracking_updated = True
        # prev_region.curr_frame_index = self.curr_frame_index - 1
        # prev_region.next_prev_regions = self.next_prev_regions
        #
        # self.next_prev_regions[self.curr_frame_index - 1] = prev_region
        #
        # if self.curr_frame_index > 2*self.k:
        #     self.next_prev_regions.pop(self.curr_frame_index - 2*self.k)


    def get_i_region(self, i):
        if -self.k < i < self.k:
            return self.next_prev_regions[self.curr_frame_index + i]
        else:
            print('{} is larger that the saved {} frames'.format(i, self.k))
            return None
