from Interface.Tracker import Tracker
import cv2
import numpy as np
import matplotlib.path as mplPath

alpha = 0.6
MIN_MATCH_COUNT = 20

class BruteForceFeatureMatchingTracker(Tracker):

    def __init__(self, feature="ORB", shape=(1000, 1000)):
        super().__init__()
        self.id = id(self)
        self.features_descriptors = []
        self.features_points = []
        self.shape = shape

        if feature == "ORB":
            self.feature_detector = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
        elif feature == "SIFT":
            self.feature_detector = cv2.xfeatures2d.SIFT_create()


    # === init_frame ===
    def init_frame(self, ad_region):

        frame = ad_region.original_frame

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        mask = cv2.drawContours(mask, [np.asarray(ad_region.region_corners, dtype=int)], -1, 1, thickness=cv2.FILLED)

        kpoints, descriptors = self.feature_detector.detectAndCompute(frame, mask)

        bbox = np.asarray(ad_region.region_corners)

        w, h = self.shape
        src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        dst_pts = bbox.reshape((-1, 1, 2))

        M, mask1 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        src_feature_points = []
        feature_descriptors = []
        for i in range(len(kpoints)):
            p = kpoints[i]
            src_feature_points.append(p.pt)
            feature_descriptors.append(descriptors[i])

        src_feature_points = np.asarray(src_feature_points).reshape(-1, 1, 2)
        dst_feature_points = cv2.perspectiveTransform(src_feature_points, M)
        feature_descriptors = np.asarray(feature_descriptors)

        self.features_descriptors.append(feature_descriptors)
        self.features_points.append(dst_feature_points)

        return ad_region

    # === process_frame ===
    def process_frame(self, ad_region):

        frame = ad_region.original_frame

        if len(self.features_descriptors) == 0:
            return self.init_frame(ad_region)

        kpoints, descriptors = self.feature_detector.detectAndCompute(frame, None)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.knnMatch(self.features_descriptors[-1], descriptors, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < alpha * n.distance:
                good.append(m)

        # print('found {} good matches'.format(len(good)))
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.features_points[-1][m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            M, mask1 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            h, w = self.shape

            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, H)

            bbPath = mplPath.Path(np.int32(dst).reshape(-1, 2))

            _feature_descriptors = []
            src_feature_points = []
            for i in range(len(kpoints)):
                p = kpoints[i]
                if bbPath.contains_point(p.pt):
                    src_feature_points.append(p.pt)
                    _feature_descriptors.append(descriptors[i])

            src_feature_points = np.asarray(src_feature_points).reshape(-1, 1, 2)
            _dst_feature_points = cv2.perspectiveTransform(src_feature_points, M)
            _feature_descriptors = np.asarray(_feature_descriptors)

            dst_feature_points = _dst_feature_points
            feature_descriptors = _feature_descriptors

            if dst_feature_points is None:
                dst_feature_points = np.array([])

            self.features_descriptors.append(feature_descriptors)
            self.features_points.append(dst_feature_points)

            # print(dst)
            ad_region.region_corners = dst

            ad_region.tracking_updated = True

            return ad_region

        print('no good feature where matched')

        return ad_region

