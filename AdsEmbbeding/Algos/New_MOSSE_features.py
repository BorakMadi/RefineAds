# -*- coding: utf-8 -*-

# !/usr/bin/env python

# Python 2/3 compatibility
# from __future__ import print_function
import sys

# from VideoFileDecoder import VideoFileDecoder
# from Encoder import Encoder
# import myUtilis

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import sys
import numpy as np
import cv2 as cv

# PARAMETERS

method = cv.TM_CCORR_NORMED
methods = ['cv.TM_CCORR_NORMED']

sqdiff_th = 0.02
cc_th = 0.99


def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand() - 0.5) * coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5) * coef
    c = (w / 2, h / 2)
    T[:, 2] = c - np.dot(T[:2, :2], c)
    return cv.warpAffine(a, T, (w, h), borderMode=cv.BORDER_REFLECT)


# divide the A on B as element wise !
def divSpec(A, B):
    Ar, Ai = A[..., 0], A[..., 1]
    Br, Bi = B[..., 0], B[..., 1]
    C = (Ar + 1j * Ai) / (Br + 1j * Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C


eps = 1e-5

flag_show = False


def getpolySubPix(frame, mask):
    if (flag_show == True):
        cv.imshow("The mask poly", mask)
        cv.waitKey(0)

    mask = mask.astype(np.bool)
    out = np.zeros_like(frame)
    out[mask] = frame[mask]

    if (flag_show == True):
        cv.imshow("The masked image ", out)
        cv.waitKey(0)

    return out


#### This correction can help us find the Homography for the orietation of the object 
def perspective_correction(im_src, pts_src, size):
    #################
    # Source Points  
    #################

    #    We get this from the first 4-polygon mask !

    #################
    # Dest Points
    #################

    # Destination image

    #    pts_dst = np.array([ 0 ,0],[ 0 ,size[1]-1],[ size[0]-1 ,size[1]-1],[size[0]-1 ,0])
    #

    pts_dst = np.array([
        [0, 0],
        [size[0] - 1, 0],
        [size[0] - 1, size[1] - 1],
        [0, size[1] - 1]], dtype="float32")

    # Calculate the homography
    h_inv, status_inv = cv.findHomography(pts_src, pts_dst)

    # Warp source image to destination
    im_dst = cv.warpPerspective(im_src, h_inv, size)
    return im_dst


def draw_image_polygon(img_new, pts_prev, masked=None):
    cv.polylines(img_new, np.int32([pts_prev]), True, (0, 255, 255), 3, cv.LINE_AA)  # this is color
    return img_new


def order_points(pts):
    pts = pts.reshape((-1,2))
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    # print('pts',pts.shape)
    # print('rect shape',rect.shape)
    s = pts.sum(axis=1)
    # print('s shape ',s.shape)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, ObjShape=None, offset=None, cropped=True):
    # obtain a consistent order of the points and unpack them
    # individually

    if offset is None:
        offset = [0, 0]
    pts = np.array(pts, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    if ObjShape is None:
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        dst += np.full(dst.shape, offset)

    else:
        #         print(ObjShape)
        if len(ObjShape) == 3:
            maxWidth, maxHeight, _ = ObjShape
        else:
            maxWidth, maxHeight = ObjShape
        dst = np.array([
            [0, 0],
            [maxWidth, 0],
            [maxWidth, maxHeight],
            [0, maxHeight]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    H_t = cv.getPerspectiveTransform(dst, rect)
    if cropped is True:
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    else :
        warped = cv.warpPerspective(image, M, (image.shape[1],image.shape[0]))

    # return the warped image
    return warped, dst, H_t


# rectangle to 4 points
def bbox_4points(b):
    rect_point = (b[0], b[1]), (b[0] + b[2], b[1]), (b[0] + b[2], b[1] + b[3]), (b[0], b[1] + b[3])

    return rect_point


def split_img(ref_img, source_img, numsplit, psr_th, th_msk):
    # ref_img : the reference image, the rectified
    # source_img : it may have occluded parts

    mask = np.zeros((ref_img.shape[0], ref_img.shape[1]))

    #        cv.imshow('sImage', source_img)
    #        print('sImage shape',source_img.shape)
    #        cv.imshow('Image', ref_img)
    #        print('Image shape',ref_img.shape)
    #        cv.waitKey(0)
    #
    #
    #

    r_len = int(ref_img.shape[0] // np.sqrt(numsplit))  # row
    c_len = int(ref_img.shape[1] // np.sqrt(numsplit))  # cols
    #        print('size of mini-image' , (r_len,c_len))

    cnt = 1
    #        print(range(0,int(np.sqrt(numsplit))))
    for i in range(0, int(np.sqrt(numsplit))):
        for j in range(0, int(np.sqrt(numsplit))):
            start_r = i * r_len
            start_c = j * c_len
            cnt = cnt + 1
            #                 crop_ref_img = ref_img[start_r:r_len -1 +start_r, start_c:start_c+c_len -1]
            #                 crop_source_img = source_img[start_r:r_len -1 +start_r, start_c:start_c+c_len -1]
            crop_ref_img = ref_img[start_r:r_len + start_r, start_c:start_c + c_len]
            crop_source_img = source_img[start_r:r_len + start_r, start_c:start_c + c_len]

            #                 cv.imshow('crop_ref_img', crop_ref_img)
            #                 print('crop_ref_img shape',crop_ref_img.shape)
            #                 cv.imshow('crop_source_img', crop_source_img)
            #                 print('crop_source_img shape',crop_source_img.shape)
            #                 cv.waitKey(0)
            #
            #                 print("The shape of Image ",crop_source_img.shape)
            if (len(crop_ref_img.shape) == 3):
                crop_ref_img_gray = cv.cvtColor(crop_ref_img, cv.COLOR_RGB2GRAY)
            else:
                crop_ref_img_gray = crop_ref_img
            if (len(crop_source_img.shape) == 3):
                crop_source_img_gray = cv.cvtColor(crop_source_img, cv.COLOR_RGB2GRAY)
            else:
                crop_source_img_gray = crop_source_img

            ##################################

            for meth in methods:
                method_l = eval(meth)
                res = cv.matchTemplate(crop_source_img_gray, crop_ref_img_gray, method_l)
                if method_l in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                    if (max_val < sqdiff_th):

                        mask[start_r:r_len + start_r, start_c:start_c + c_len] = 1
                    else:
                        if (crop_ref_img.shape[0] <= 2 or crop_ref_img.shape[1] <= 2):
                            mask[start_r:r_len + start_r, start_c:start_c + c_len] = 0
                        else:

                            crop_mask = split_img(crop_ref_img, crop_source_img, numsplit, psr_th, th_msk)
                            mask[start_r:r_len + start_r, start_c:start_c + c_len] = crop_mask
                else:
                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                    if (max_val > cc_th):
                        mask[start_r:r_len + start_r, start_c:start_c + c_len] = 1
                    else:
                        if (crop_ref_img.shape[0] <= 2 or crop_ref_img.shape[1] <= 2):
                            mask[start_r:r_len + start_r, start_c:start_c + c_len] = 0
                        else:
                            crop_mask = split_img(crop_ref_img, crop_source_img, numsplit, psr_th, th_msk)
                            mask[start_r:r_len + start_r, start_c:start_c + c_len] = crop_mask
    return mask


class MOSSE:

    def __init__(self, frame, poly_polys, psr_th, h_transform=np.identity(3), std_norm=10):

        print('Start Init ')
        # polygon's points we get these points from User/MaskRCNN with shape list of pairs/tuple
        # Example [(209, 122), (380, 138), (381, 228), (264, 237)]

        self.window_rect = []
        self.polygon_points = poly_polys
        self.FrameHeight, self.FrameWith, _ = frame.shape

        # order points clockwise  & construct rectified image
        # the function habe ability to return the transform/homography too.
        # rect : rectagle point of the image (topl_left) and (bottom_right) style
        # recti_img : rectified image
        recti_img, rect, _ = four_point_transform(frame, poly_polys)

        ########## rectified section ##########
        # rectangle is  (topl_left) and (bottom_right) style
        x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[2][0], rect[2][1]
        w, h = map(cv.getOptimalDFTSize, [x2 - x1, y2 - y1])
        self.img_first = cv.resize(recti_img, (w, h))
        self.point_rectangle = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        self.rectangle = [x1, y1, w, h]

        recti_img_whole, _, _ = four_point_transform(frame, poly_polys, None, poly_polys[0],False)
        # cv.imshow("Rectified Image", recti_img_whole)
        # cv.waitKey(0)

        ######### Calculation after rectified ##########
        x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
        self.pos = x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)  # mid point of  the rectangle
        self.size = w, h
        self.th_psr_th = psr_th
        img = cv.getRectSubPix(recti_img, (w, h), (x, y))  # get sub image from whole image
        #        print('The shape of img ',img.shape)
        self.win = cv.createHanningWindow((w, h), cv.CV_32F)  # creatine Hanning Window
        g = np.zeros((h, w), np.float32)
        g[h // 2, w // 2] = 1
        g = cv.GaussianBlur(g, (-1, -1), std_norm)  # create Gaussian filter  with - dev = 2.0
        g /= g.max()  # normalize the Gaussian with

        self.G = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)  # compute dft of Gaussian :D
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)

        ####### Before Rectified #########
        self.pos_oriented = cv.perspectiveTransform(np.float32(np.array([[self.pos]])),
                                                    h_transform)  ## now its transform
        self.init_pos = self.pos
        # try some pertuabs of img
        if (len(frame.shape) == 3):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for _i in xrange(128):
            a = self.preprocess(rnd_warp(img))  # try some pertuabs , like rotation ,ilumination
            A = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)  # dft of A
            self.H1 += cv.mulSpectrums(self.G, A, 0, conjB=True)  # the numerator
            self.H2 += cv.mulSpectrums(A, A, 0, conjB=True)  # is denominator or spectrum of Energy !0

        img_ = np.zeros_like(frame)
        if (len(img_.shape) == 3):
            img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)

        self.update_kernel()
        self.update(frame)
        print('End Init ')

    # update is
    def update(self, frame, h_transform=np.identity(3), rate=0.125):

        #####################
        #  Oriented Method
        # ####################
        #        self.pos_oriented =  np.float32(np.array([[self.init_pos]]))
        oriented_position = cv.perspectiveTransform(self.pos_oriented, h_transform)  ## now its transform
        self.pos = tuple(oriented_position[0][0].tolist())
        (x, y), (w, h) = self.pos, self.size  # size of the rectangle + or it's mid point

        self.h_trans = h_transform
        self.warpmaskOcc = None

        #        cv.imshow('raw frame ',  frame )
        #        cv.waitKey(0)

        # ======== build the masks========= ###
        # build mask as filling 4-ver polygon 1 where our desired regions and 0 where there are nots
        # self.mask the mask (polygon) of the first frame 
        self.mask = np.zeros((frame.shape[0:2]))
        xys = np.array(self.polygon_points, np.int32)
        cv.fillConvexPoly(self.mask, xys, 1)

        #        cv.imshow('Init Mask',   self.mask )
        #        cv.waitKey(0)

        # ====== build the mask  but rectified  ==== #
        # this is the projection of the mask above, to rectifiy it :D 
        # self.maskrect the mask (rectified ) of the first frame 
        self.maskrect = np.zeros((frame.shape[0:2]))
        xys = np.array(self.point_rectangle, np.int32)
        cv.fillConvexPoly(self.maskrect, xys, 1)

        #        cv.imshow('Rectangle Mask',   self.maskrect )
        # ====== Calc the dst points incoming frame  ==== #
        # polygon_points : the points of first frame polygon
        # pts_scr : the same value as polygon_points but different repesent to work with perspectiveTransform well
        # pts_dst : after apply the homography transformation :D we get these points
        pts_scr = np.array(self.polygon_points, dtype=np.float32)
        pts_scr = np.array([pts_scr])
        pts_dst = cv.perspectiveTransform(pts_scr, h_transform)

        self.pt_dst = pts_dst
        #        self.warpmask = self.mask.copy()

        # ====== build the mask FOR INCOMING IMAGE/FRAME ==== #
        self.warpmask = cv.warpPerspective(self.mask, h_transform, (self.mask.shape[1], self.mask.shape[0]))
        #        cv.imshow('Warppu Mask',  self.warpmask)

        # ====== The masked frame for the ROI =========#
        # Here we have to get the masked region only , and zero elsewhere .
        # we apply the mask 'self.warpmask' to do the job 
        masked_img = getpolySubPix(frame, self.warpmask)

        # ====== Finally rectified the region  =========#
        rectifed_img, rect, H_inv = four_point_transform(masked_img, pts_dst[0], (w, h))
        ####  can make probems later !
        #        h, w,_ = self.img_first.shape
        #        rectifed_img = cv.resize(rectifed_img,(w,h))
        #        print('after resize rectifed_img', rectifed_img.shape)
        self.last_img = img = rectifed_img

        if (len(frame.shape) == 3):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = self.preprocess(img)  # get transformations from image

        # correlation is main function here :D , (dx,dy) is movement of the rectangle
        # last_resp the response map

        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        # print("The PSR Value ",self.psr)
        self.good = self.psr > self.th_psr_th

        img_gray = cv.cvtColor(self.img_first, cv.COLOR_RGB2GRAY)

        #        print('the shape of rectifed' , rectifed_img.shape)
        #        print('the shape of our template ' , img_gray.shape)
        if (len(rectifed_img.shape) == 3):
            rectifed_img_gray = cv.cvtColor(rectifed_img, cv.COLOR_RGB2GRAY)
        else:
            rectifed_img_gray = rectifed_img
        ##################################

        # calc the  numerator and deno
        A = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv.mulSpectrums(A, A, 0, conjB=True)

        # learning step,
        # calc pervious and curr H1 , H2
        self.H1 = self.H1 * (1.0 - rate) + H1 * rate
        self.H2 = self.H2 * (1.0 - rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv.idft(self.H, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        h, w = f.shape
        f = np.roll(f, -h // 2, 0)
        f = np.roll(f, -w // 2, 1)
        kernel = np.uint8((f - f.min()) / f.ptp() * 255)
        resp = self.last_resp
        resp = np.uint8(np.clip(resp / resp.max(), 0, 1) * 255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    # draw the retangle and PSR value , or crossed-rectangle when there is not enough PSR

    def draw_state(self, vis):

        if self.good:
            self.draw_rectangle(vis, True)
            # lets draw the oriented rectangles
            return draw_image_polygon(vis, self.pt_dst)

        else:
            self.draw_rectangle(vis, False)

    def draw_rectangle(self, vis, flag):

        list1, list2 = np.asarray(self.pt_dst[0, :, 0], dtype=np.int32).tolist(), np.asarray(self.pt_dst[0, :, 1],
                                                                                             dtype=np.int32).tolist()
        min_x = min(list1)
        max_x = max(list1)
        min_y = min(list2)
        max_y = max(list2)
        x1, y1, x2, y2 = min_x, min_y, max_x, max_y
        # cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if flag is True:
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        else:
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv.line(vis, (x2, y1), (x1, y2), (0, 0, 255))

        return vis

    #  preprocess  of img to give stable kernel !
    def preprocess(self, img):
        img = np.log(np.float32(img) + 1.0)
        img = (img - img.mean()) / (img.std() + eps)
        return img * self.win

    def correlate(self, img):

        # get the result of  img * kernel ! 
        C = cv.mulSpectrums(cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)

        # response map of the result
        resp = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        h, w = resp.shape

        # find the maximum value -
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)

        side_resp = resp.copy()
        cv.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)

        # calc  the PSR value 
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval - smean) / (sstd + eps)

        return resp, (mx - w // 2, my - h // 2), psr

    # recompute the kernel after getting new image !
    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[..., 1] *= -1

