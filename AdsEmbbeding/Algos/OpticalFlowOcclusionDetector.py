from Interface.OcclusionDetector import OcclusionDetector
import numpy as np
import cv2

SHOW_RESULTS = False


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


class OpticalFlowOcclusionDetector(OcclusionDetector):

    def __init__(self, mask_size=(500, 500), margin=20):

        super().__init__()
        self.mask_size = mask_size
        self.margin = margin

        self.curr_mask = np.ones(self.mask_size)

        self.background_mask = np.zeros((self.mask_size[0]+2*margin, self.mask_size[1]+2*margin))
        self.background_mask[self.margin:self.margin+self.mask_size[0], self.margin:self.margin+self.mask_size[1]] = np.ones(self.mask_size)


        self.tracked_image = None
        self.curr_optical_flow = None

        # === update_mask ===
    def update_mask(self, ad_region):

        H = ad_region.get_homography(self.mask_size)

        corners_def = np.asarray([(-self.margin, -self.margin),
                                  (-self.margin, self.mask_size[1]+self.margin),
                                  (self.mask_size[0]+self.margin, self.mask_size[1]+self.margin),
                                  (self.mask_size[0]+self.margin, -self.margin)], dtype=np.float32).reshape(-1, 1, 2)


        t_corners_def = cv2.perspectiveTransform(corners_def, H)


        new_corners = t_corners_def

        w, h = self.background_mask.shape
        src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        dst_pts = np.array(new_corners, np.float32)
        M, mask = cv2.findHomography(dst_pts, src_pts)


        frame = ad_region.original_frame

        # rectify
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_gray = cv2.warpPerspective(frame_gray, M, self.background_mask.shape)
        OpticalFlowOcclusionDetector.py
        if self.tracked_image is None:
            self.tracked_image = frame_gray


        # Calculate the optical flow of the region (along with the margin) in the current frame
        self.curr_optical_flow = cv2.calcOpticalFlowFarneback(self.tracked_image, frame_gray,
                                                              self.curr_optical_flow, pyr_scale=0.5, levels=5,
                                                              winsize=13,
                                                              iterations=30, poly_n=5, poly_sigma=1.1, flags=0)

        flow = np.asarray(np.rint(self.curr_optical_flow), dtype=np.int32)

        if SHOW_RESULTS:
            cv2.imshow('region_gray', frame_gray)
            cv2.imshow('prev region', self.tracked_image)

            hsv = np.zeros((frame_gray.shape[0], frame_gray.shape[1], 3), dtype=np.uint8)
            hsv[..., 1] = 255

            mag, ang = cv2.cartToPolar(self.curr_optical_flow[..., 0], self.curr_optical_flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow("colored flow", bgr)

        # self.curr_optical_flow = self.flow.processFrame(region)

        for x in range(flow.shape[0]):
            for y in range(flow.shape[1]):
                of = flow[x][y]
                p = np.asarray([x, y])
                # dst = np.asarray([x+of[0], y+of[1]])
                if self.margin < x + of[0] < self.background_mask.shape[0]-self.margin and\
                        self.margin < y + of[1] < self.background_mask.shape[1]-self.margin:
                    self.background_mask[x + of[0]][y + of[1]] = self.background_mask[x][y]


        self.curr_mask = self.background_mask[self.margin:self.margin+self.mask_size[0], self.margin:self.margin+self.mask_size[1]]

        mask = cv2.warpPerspective(self.curr_mask, H, (frame.shape[1], frame.shape[0]))

        if SHOW_RESULTS:
            cv2.imshow('background mask', 255 * self.background_mask)
            cv2.imshow('mask', 255 * self.curr_mask)
            cv2.imshow('rect mask', 255 * mask)
            cv2.waitKey(1)



        ad_region.mask = self.curr_mask
        ad_region.mask_updated = True

        self.tracked_image = frame_gray

        return ad_region
