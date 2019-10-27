import numpy as np
import cv2

from Algos.FeatureExtraction import FeatureExtraction
from Algos import New_MOSSE_features as mos
from Algos.New_MOSSE_features import four_point_transform

alpha = 0.7
MIN_MATCH_COUNT = 20
## Paramters of algorithim of Homography Estimation 
algo_Method = cv2.RHO
threshold = 3  # default is 3.0 is used for RANSAC and RHO only
MaxIter = 2020  # default number of iterations to run
confidence_t = 0.995  # the confidence level
GOOD_MATCH_PERCENT = 1  ## our solutions build from data set !

## MOSSE Paramters 
psr_threth = 8.0

SHOW_RESULTS = True


# convert bbox to 4 points --> [x y w h ]  to 4 points
def bbox_4points(b):
    rect_point = (b[0], b[1]), (b[0] + b[2], b[1]), (b[0] + b[2], b[1] + b[3]), (b[0], b[1] + b[3])

    return rect_point


def draw_image_polygon(img_new, pts_prev, masked=None):
    cv2.polylines(img_new, np.int32([pts_prev]), True, (0, 255, 255), 3, cv2.LINE_AA)  # this is color
    return img_new


class MOSSEOrientedTracker:

    def __init__(self, shape=(1000, 1000), feature="ORB"):

        self.id = id(self)
        #        self.adRegion = adRegion
        self.shape = shape
        #        self.adShape = adRegion.get_adShape()
        #
        # we will use the default parameters but we can change :D 
        # self.TkFeatExtr = FeatureExtraction()
        self.TkFeatExtr = FeatureExtraction(feature, feature)
        # correlation filters
        self.corFilter = []
        self.isOccluded = False
        self.isSuccessive = False

        self.FirstFrameForExtract = None
        self.p_kp = None
        self.p_desc = None
        self.initpoints = None

        # === init_frame ===
    # the Homography Estimation Paramters
    def init_frame(self, ad_region, algo_Method=cv2.RANSAC, threshold=3, maxIters_t=2020, confidence_t=0.995):

        frame = ad_region.original_frame
        poly_pts = np.asarray(ad_region.region_corners).reshape(-1,2)
        frame_cop = frame.copy()

        # ======== let's fix aspects of the ad =========
        rectifed_img, rect, _ = four_point_transform(frame_cop, poly_pts)
        w, h = rect[2][0], rect[2][1]
        # cv2.imshow("Rect imag", rectifed_img)
        # cv2.waitKey(0)

        #        ========= Homography Estimation ==============
        # extract the reference keypoints and descriptors using bbox as mask
        self.TkFeatExtr.init_frame(frame_cop, poly_pts)
        # cv2.imshow("the mask", self.TkFeatExtr.Extmask)
        # cv2.waitKey(0)
        # cv2.imshow("the points", self.TkFeatExtr.frameWKeyPoints)
        # cv2.waitKey(0)

        # save the feature points with first frame
        self.FirstFrameForExtract = self.TkFeatExtr.frameWKeyPoints
        if self.isSuccessive is False:
            self.p_kp, self.p_desc = self.TkFeatExtr.kp_feats, self.TkFeatExtr.des_feats

        self.initpoints = poly_pts
        # src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        src_pts = np.float32(poly_pts)
        # We can use bbox the polygon and the bounding box       
        dst_pts = np.array(poly_pts, np.float32)

        # we can make it more simple :D 
        M, mask1 = cv2.findHomography(dst_pts, src_pts, algo_Method, ransacReprojThreshold=threshold, maxIters=MaxIter,
                                      confidence=confidence_t)
        H, mask = cv2.findHomography(src_pts, dst_pts, algo_Method, ransacReprojThreshold=threshold, maxIters=MaxIter,
                                     confidence=confidence_t)

        # ok let's take the feature points inside the container !
        # we can do this using mask wth opencv function !
        img_polygoned = draw_image_polygon(frame_cop, dst_pts)
        # cv2.imshow('Polygoned Image',img_polygoned)
        # cv2.waitKey(0)
        self.curr_H = np.round(H)

        #        =========== Correlation Filters ================
        self.corFilter = mos.MOSSE(frame, poly_pts, psr_threth, H)

        return ad_region

    # === process_frame ===
    def process_frame(self, ad_region):

        if len(self.TkFeatExtr.des_feats) == 0:
            return self.init_frame(ad_region)

        frame = ad_region.original_frame.copy()
        #        poly_pts =  ad_region.region_corners

        ###### Estiamte Global Homography to estimate 2 the relation between 2 views !
        if self.isSuccessive is False:

            drawImpointsSecond, kp2, des2 = self.TkFeatExtr.detect_and_compute(frame)
            # vis = np.concatenate((self.FirstFrameForExtract, drawImpointsSecond), axis=1)
            # cv2.imshow("imag with Keypoints with Mask", vis)
            # cv2.waitKey(0)

            # Match the features
            # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            # matches = bf.knnMatch(self.p_desc, des2, k=2)  # typo fixed
            # # Apply ratio test

            matches = self.TkFeatExtr.matchFeatures(des2)

            good = []
            for m, n in matches:
                if m.distance < alpha * n.distance:
                    good.append(m)

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([self.p_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                Global_H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 5.0)
                self.curr_H = Global_H

            #           =========== Correlation Filters ================
            self.corFilter.update(frame, self.curr_H)
            self.isOccluded = not self.corFilter.good
            print("is it Occlued",  self.corFilter.good)
            pts = np.float32(self.initpoints).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, self.curr_H)

            img_polygoned = draw_image_polygon(frame, dst)

            ad_region.update_corners(dst)

            return ad_region


        else:

            print('no good feature where matched')

        return ad_region


#


if __name__ == '__main__':
    print('Hi World')
