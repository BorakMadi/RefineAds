
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os,sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from AdDetection.RCF_Utility import RCF_BoundaryDetector as RCFBD

SHOW_RESULTS = True

eps = 1e-5
def normalize(points):
    for row in points:
        row /= points[-1]
    return points


def order_points(pts):
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
def four_point_transform(image, pts, ObjShape=None):
    # obtain a consistent order of the points and unpack them
    # individually

    pts = np.array(pts, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    if (ObjShape is None):
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
    else:
        #         print(ObjShape)
        if (len(ObjShape) == 3):
            maxWidth, maxHeight, _ = ObjShape
        else:
            maxWidth, maxHeight = ObjShape
        dst = np.array([
            [0, 0],
            [maxWidth, 0],
            [maxWidth, maxHeight],
            [0, maxHeight]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    src_dst_t = cv2.getPerspectiveTransform(rect, dst)
    dst_src_t = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, src_dst_t, (maxWidth, maxHeight))

    # return the warped image
    return warped, dst, dst_src_t,src_dst_t


def getpolySubPix(frame, mask):

    mask = mask.astype(np.bool)
    out = np.zeros_like(frame)
    out[mask] = frame[mask]
    return out

class TemplateMatchingOcclusionDetector():
    def __init__(self):

        self.mask = []
        self.methodTM_OccIndicator = cv2.TM_CCORR_NORMED
        self.methodTM_OccBuilder = cv2.TM_CCORR_NORMED
        self.methodTM_Validation = cv2.TM_SQDIFF_NORMED
        self.sqdiff_th = 0.1
        self.cc_th = 0.99
        self.isFirst = True
        self.Method = "ColorTemplate"
        self.TemplateValues = []
        self.skipcnt =0
        self.TemplateFrequency = []
        self.outVideo = []
        self.OutVideMasked=[]
        self.refine_mask_build = np.ones((100,100))
        self.rcf_model = RCFBD.RCF_BoundaryOcclusionBoundaryDetector()
    def first_frame(self,ad_region):

        # polygon's points for the first frame
        self.polygon_points = np.asarray(ad_region.region_corners).reshape(-1,2)

        # get the first frame
        first_frame = ad_region.original_frame
        self.FrameHeight, self.FrameWith, _ = first_frame.shape

        # print(self.polygon_points)
        # order points & construct rectified image + crop !
        recti_img, rect, _,_ = four_point_transform(first_frame, self.polygon_points)

        ########## rectified section ##########
        # rectangle is  (topl_left) and (bottom_right) style
        x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[2][0], rect[2][1]
        w, h = map(cv2.getOptimalDFTSize, [x2 - x1, y2 - y1])
        self.point_rectangle = [[x1, y1], [x1, h], [w, h], [w, y1]]
        self.rectangle = [x1, y1, w, h]
        self.size = w, h
        self.isFirst = False


        self.TemplateRefrence  = cv2.resize(recti_img, (w, h)) # Example : (120, 160, 3)
        height, width, layers = self.TemplateRefrence.shape
        # print("the shape ColorImage is ", self.TemplateRefrence.shape," while the optimized is ", (h,w))

        # self.outVideo = cv2.VideoWriter('video.avi', -1, 1, (2 * width, height))
        # self.OutVideMasked= cv2.VideoWriter('video.avi', -1, 1, (2 * self.FrameWith, self.FrameHeight))
        self.TemplateFirstRefrenceGray = cv2.cvtColor(self.TemplateRefrence, cv2.COLOR_RGB2GRAY)  ## the template with gray colors !
        self.TemplateRefrenceGray = self.TemplateFirstRefrenceGray.copy()
        self.init_psr, self.LastFrameFrequency, self.FirstFrameFrequency = self.correlate(np.float32(self.TemplateFirstRefrenceGray),
                                                                                                 np.float32( self.TemplateFirstRefrenceGray))


        # === update_mask ===
    def update_mask(self,ad_region,isOccluded=False):

        if(self.isFirst is True):
            return self.first_frame(ad_region)

        frame = ad_region.original_frame
        # ====== build the mask  but rectified  ==== #
        # this is the projection of the mask above, to rectifiy it :D
        # self.maskrect the mask (rectified ) of the first frame
        self.maskrect = np.zeros((frame.shape[0:2]))
        xys = np.array(self.point_rectangle, np.int32)
        cv2.fillConvexPoly(self.maskrect, xys, 255)
        self.MaskNonTransform = np.ones(self.size)

        # xys = np.array(ad_region.region_corners, np.int32)
        # cv2.fillConvexPoly(self.mask, xys, 1)

        ## self.maskrect the same as frame size with rectified and translated to (0,0) template.
        # cv2.imshow("Mask rect ", self.maskrect)
        # cv2.waitKey(0)


        # ======== build the masks========= ###
        # build mask as filling 4-ver polygon 1 where our desired regions and 0 where there are nots
        # self.mask the mask (polygon) of the first frame

        self.mask = np.zeros((frame.shape[0:2]))
        xys = np.array(ad_region.region_corners, np.int32)
        cv2.fillConvexPoly(self.mask, xys, 1)
        # Mask ad_region.region_corners and filled with 1
        # cv2.imshow("Mask changed", self.mask)
        # cv2.waitKey(0)

        self.umask = 255-255*self.mask
        pts_scr = np.array(self.point_rectangle, dtype=np.float32)
        pts_dst =  np.array(ad_region.region_corners, dtype=np.float32)
        h_transform, mask1 = cv2.findHomography(pts_scr, pts_dst,cv2.RANSAC, 5.0)
        self.warpmask = cv2.warpPerspective(self.maskrect, h_transform, (self.mask.shape[1], self.mask.shape[0]))


        # cv2.imshow('Warppu Mask', self.warpmask)
        # cv2.waitKey(0)

        # ====== The masked frame for the ROI =========#
        # Here we have to get the masked region only , and zero elsewhere .
        # we apply the mask 'self.warpmask' to do the job
        masked_img = getpolySubPix(frame, self.warpmask)
        # cv2.imshow('masked_img ', masked_img)
        # cv2.waitKey(0)

        # ====== Finally rectified the region  =========#
        w, h = self.size
        flatten_array = pts_dst.flatten().reshape(-1,2);
        update_points = [tuple(x) for x in flatten_array.tolist()]

        # array to list of tuple
        rectifed_img, rect, H_inv,H_srcDst = four_point_transform(masked_img, update_points, (w, h))
        # cv2.imshow('Rectifiy Incoming images ', rectifed_img)
        # cv2.waitKey(1)
        self.skipcnt = self.skipcnt + 1
        if (len(rectifed_img.shape) == 3):
            rectifed_img_gray = cv2.cvtColor(rectifed_img, cv2.COLOR_RGB2GRAY)
        else:
            # cv2.waitKey(0)
            rectifed_img_gray = rectifed_img

            ##########################################
            # Correlation_NORM Template Matching
            #########################################

        # let's compute the results without care about RGB first !
        if self.Method == "ColorTemplate":
            # self.TemplateRefrenceGray the last template that not occluded
            self.nowpsr,_, self.IncomingFrameFrequency= self.correlate(np.float32(self.TemplateFirstRefrenceGray),np.float32(rectifed_img_gray))
            print("The PSR in Template Matching ",self.nowpsr)
            if (self.nowpsr/self.init_psr > 0.80) :
                print("No Occlusions")
                # self.TemplateRefrenceGray = rectifed_img_gray
                # self.init_psr = self.nowpsr
                ad_region.mask = np.ones((100, 100))
            else :
                print("there is occlusion")
                # cv2.imshow("The rectifed_img_gray", rectifed_img_gray)
                # cv2.waitKey(0)
                self.recompute_mask(self.TemplateRefrence, rectifed_img, H_inv);
                ad_region.mask = 255 - self.refine_mask_build
                # cv2.imshow("The mask the results",255 - self.refine_mask_build)
                # cv2.waitKey(0)


        # ad_region.edited_frame =self.MaskNonTransform * rectifed_img
        ad_region.mask_updated = True
        return ad_region
    def correlate(self,template, img):

        # get the result of  img * kernel !
        dft_free_template = cv2.dft(template, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft_free_template)
        # magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        # plt.subplot(121), plt.imshow(template, cmap='gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.show()
        ### Image Back ###
        # f_ishift = np.fft.ifftshift(dft_shift)
        # img_back = cv2.idft(f_ishift)
        # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        # plt.title('Image Back'), plt.xticks([]), plt.yticks([])
        # plt.imshow(img_back, cmap='gray')
        # plt.show()
        dft_fre_img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift_2 = np.fft.fftshift(dft_fre_img)
        # magnitude_spectrum_2 = cv2.magnitude(dft_shift_2[:, :, 0], dft_shift_2[:, :, 1])
        magnitude_spectrum_2 = 20 * np.log(cv2.magnitude(dft_shift_2[:, :, 0], dft_shift_2[:, :, 1]))
        # plt.subplot(121), plt.imshow(img, cmap='gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(magnitude_spectrum_2, cmap='gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.show()
        C = cv2.mulSpectrums(dft_free_template,dft_fre_img,0, conjB=True)


        # plt.imshow(C, cmap='gray')
        # plt.title('Magnitude Spectrum Multiply'), plt.xticks([]), plt.yticks([])
        # plt.show()
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        h, w = resp.shape
        # find the maximum value     -
        side_resp = resp.copy()
        minval, mval, _, (mx, my) = cv2.minMaxLoc(side_resp)
        # cv2.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        # print(side_resp)
        smean, sstd = side_resp.mean(), side_resp.std()

        # print("the max value", mval, "the min value", minval, "the smean", smean, " and the sstd", sstd)
        # print("The max location",(mx, my))
        print(" The PSR", (mval - smean) / (sstd + eps))
        #
        # # response map of the result
        # resp = cv2.idft(C, flags= cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
        # ###########
        # resp_shift = np.fft.ifftshift(dft_shift_2)
        # resp = resp_shift.copy()
        # plt.imshow(resp_shift, cmap='gray')
        # plt.title('resp Map'), plt.xticks([]), plt.yticks([])
        # plt.show()
        #
        # h, w = resp.shape
        # # find the maximum value     -
        # side_resp = resp.copy()
        #
        # # cv2.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        # minval, mval, _, (mx, my) = cv2.minMaxLoc(side_resp)
        # smean, sstd = side_resp.mean(), side_resp.std()
        #
        # print("the max value",mval,"the min value",minval,"the smean",smean," and the sstd",sstd)
        # print(" The PSR", (mval - smean) / (sstd + eps))

        PSR = (mval - smean) / (sstd + eps)

        return PSR,magnitude_spectrum,magnitude_spectrum_2
    def correlate_histogram(self,template, img):
        print('not implemented yet')
    def rotate_image(self,img, angle):
        # angle in degrees

        height, width = img.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
        return rotated_mat
    def connect_edges(self,edges_map):
        print(" not implemented yet!")
    def isNotBoundray(self,i,length):
        if i>0 and i<length-1:
                return True
        else:
                return False;
    def isNotFromContinueEdge(self,img):

        if(img.sum() >=255*3):
            return False
        else:
            return True
    def continueEdge(self,source_img):

        blurFact =3
        minValueThresh = 100
        initEdgeMap  = cv2.Canny(cv2.GaussianBlur(source_img, (blurFact, blurFact), 0), minValueThresh,
                                       2 * minValueThresh)
        # cv2.imshow("the initEdgeMap using canny src ", initEdgeMap)
        # cv2.imshow("the edge map using canny ref ", edges_ref)
        # cv2.waitKey(0)



        #### Let try edge detection using Canny with different paramters
        for blurFact in [3]:
            for minValueThresh in [20]:
                supportEdgeMap = cv2.Canny(cv2.GaussianBlur(source_img, (blurFact, blurFact), 0), minValueThresh,
                                       2 * minValueThresh)
                # cv2.imshow("the supportEdgeMap  using canny src ", supportEdgeMap)
                # cv2.waitKey(0)
        conc1 =  np.concatenate((initEdgeMap,supportEdgeMap), axis=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        supportEdgeMap = cv2.morphologyEx(supportEdgeMap, cv2.MORPH_CLOSE, kernel)




        # cv2.imshow(" The Edges of Canny with morphylogical ", supportEdgeMap)
        # cv2.waitKey(0)
        # cv2.imwrite('CannyMask.png', supportEdgeMap)

        (h, w) = source_img.shape[0:2]
        for i in range(0, h):
            for j in range(0, w):

                if(initEdgeMap[i,j] ==255):

                    if(self.isNotBoundray(i,h) and self.isNotBoundray(j,w)):


                            #let's if one adjacent exist only and pixel is not boundray

                            if(self.isNotFromContinueEdge(initEdgeMap[i-1:i+2,j-1:j+2])):

                                print("change")
                                print("the index is i,j",i,j)
                                print(  initEdgeMap[i-1:i+2,j-1:j+2] )
                                print(supportEdgeMap[i-1:i+2,j-1:j+2])
                                initEdgeMap[i-1:i+2,j-1:j+2] = supportEdgeMap[i-1:i+2,j-1:j+2]
                                # cv2.waitKey(0)

        conc2 = np.concatenate((initEdgeMap, supportEdgeMap), axis=1)
        conc3 = np.concatenate((conc1, conc2), axis=0)

        cv2.imshow("all steps for algorthim", conc3)
        cv2.waitKey(0)
    def NonFittingMatch(self,template,patch):
        methods = ['cv2.TM_CCORR_NORMED']
        for i in range(len(methods)):
           matching = cv2.matchTemplate(patch, template, eval(methods[i]))
           print(np.max(matching))
        return np.max(matching)
    def FittingMatch(self,patch1, patch2):
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()

        if stds == 0:
            return 0
        else:
            product /= stds
            print("product",product)
            return product
    def MatchingMask(self,ref_img_gray, source_img_gray):

        d = 3
        sh_row, sh_col = ref_img_gray.shape[0:2]

        correlation = np.ones(source_img_gray.shape[0:2])
        isFit = True  # env or fitting matching
        for d in [11]:
            field = d // 2
            if isFit is False:
                new_source_img_gray = np.pad(source_img_gray, (field, field), 'constant')

            for i in range(d, sh_row - (d + 1)):
                for j in range(d, sh_col - (d + 1)):
                    template = ref_img_gray[i - d: i + d + 1, j - d:j + d + 1]

                    if isFit is True:
                        patch = source_img_gray[i - d:i + d + 1, j - d:j + d + 1]
                        correlation[i, j] = self.FittingMatch(template, patch)
                    else:
                        # not finished yet !
                        i_source = i + field  # same point in template
                        j_source = j + field
                        patch = new_source_img_gray[i_source - d - field:i_source + d + field + 1,
                                j_source - d - field: j_source + d + field + 1]
                        correlation[i, j] = correlation[i, j] + self.NonFittingMatch(template, template)


            # We need to think in better way !

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            correlation = cv2.morphologyEx(correlation, cv2.MORPH_OPEN, kernel)


        return correlation


    def recompute_mask(self, ref_img, source_img, h_transform):



        # ref_img : the reference image, the rectified
        # source_img : it may have occluded parts

        #### start save image #####
        height, width, layers = source_img.shape
        self.acc_mask = np.zeros((ref_img.shape[0], ref_img.shape[1]))
        self.heatmap = np.zeros((ref_img.shape[0], ref_img.shape[1]))

        ################
        # Part 1 : Building Intial Occlusion Estimation Using Template Matching
        ###############

        if(len(ref_img.shape)==3):
            ref_img_gray = cv2.cvtColor(ref_img,cv2.COLOR_RGB2GRAY)
        else:
            ref_img_gray = ref_img.copy()

        if (len(source_img.shape) == 3):
            source_img_gray = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
        else:
            source_img_gray = source_img.copy()

        matchingImgMaskCorr = self.MatchingMask(ref_img_gray,source_img_gray)
        plt.imshow(matchingImgMaskCorr,cmap='gray', vmin=0, vmax=1)
        plt.show()

        # This is hyper-parameter  we can change
        ret3, th3 = cv2.threshold(matchingImgMaskCorr, 0.5,1,cv2.THRESH_BINARY)
        plt.imshow(th3, cmap='gray')
        plt.show()
        ##################################################################################


        ################
        # Part 2 : Building Intial Occlusion Estimation Using Template Matching with "Sharpning"
        ###############

        h, w = ref_img.shape[0:2]
        print(h, w)
        if h % 2 == 0: h = h + 1
        if w % 2 == 0: w = w + 1
        print(h, w)

        ref_img_ga = cv2.GaussianBlur(ref_img, (h, w), 0)
        source_img_ga = cv2.GaussianBlur(source_img, (h, w), 0)

        sharp_ref_img = cv2.addWeighted(ref_img, 1.5, ref_img_ga, -0.25, 0)
        sharp_source_img = cv2.addWeighted(source_img, 1.5, source_img_ga, -0.25, 0)
        # plt.imshow(ref_img_ga)
        # plt.show()
        # plt.imshow(sharp_ref_img)
        # plt.show()
        # plt.imshow(source_img_ga)
        # plt.show()
        # plt.imshow(sharp_source_img)
        # plt.show()
        if (len(sharp_ref_img.shape) == 3):
            sharp_ref_img_gray = cv2.cvtColor(sharp_ref_img, cv2.COLOR_RGB2GRAY)
        else:
            sharp_ref_img_gray = sharp_ref_img.copy()

        if (len(sharp_source_img.shape) == 3):
            sharp_source_img_gray = cv2.cvtColor(sharp_source_img, cv2.COLOR_RGB2GRAY)
        else:
            sharp_source_img_gray = sharp_source_img.copy()


        matchingImgMaskCorrSharp = self.MatchingMask(sharp_ref_img_gray, sharp_source_img_gray)
        plt.imshow(matchingImgMaskCorrSharp, cmap='gray', vmin=0, vmax=1)
        plt.show()

        # This is hyper-parameter  we can change
        ret3, th3_Sharp = cv2.threshold(matchingImgMaskCorrSharp, 0.5, 1, cv2.THRESH_BINARY)
        plt.imshow(th3_Sharp, cmap='gray')
        plt.show()

        #################################################################

        ################
        # Part 3 : Building Contour using Canny
        ###############
        ## Let try edge detection using Canny with different paramters
        for blurFact in [3]:
            for minValueThresh in [10]:
                edges_srcs = cv2.Canny(cv2.GaussianBlur(source_img, (blurFact, blurFact), 0), minValueThresh,
                                       2 * minValueThresh)

                edges_ref = cv2.Canny(cv2.GaussianBlur(ref_img, (blurFact, blurFact), 0), minValueThresh,
                                      2 * minValueThresh)

                edges_sharp_ref_img = cv2.Canny(cv2.GaussianBlur(sharp_ref_img, (blurFact, blurFact), 0), minValueThresh,
                                      2 * minValueThresh)

                edges_sharp_source_img = cv2.Canny(cv2.GaussianBlur(sharp_source_img, (blurFact, blurFact), 0),
                                                minValueThresh,
                                                2 * minValueThresh)
                # cv2.imshow("edges_srcs ", edges_srcs)
                # cv2.imshow("edges_ref ", edges_ref)
                # cv2.imshow("edges_sharp_ref_img ", edges_sharp_ref_img)
                # cv2.imshow("edges_sharp_source_img ", edges_sharp_source_img)
                # cv2.waitKey(0)

        # Close Gaps and make continues edges maps !


        cv2.destroyAllWindows()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        plt.imshow(edges_srcs, cmap='gray')
        plt.show()
        edges_srcs3 = cv2.morphologyEx(edges_srcs, cv2.MORPH_CLOSE, kernel)
        plt.imshow(edges_srcs3, cmap='gray')
        plt.show()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_srcs5 = cv2.morphologyEx(edges_srcs, cv2.MORPH_CLOSE, kernel)
        plt.imshow(edges_srcs5, cmap='gray')
        plt.show()




































        rcf_inputs_ref = self.rcf_model.boundary_detection(ref_img)
        rcf_inputs_src  = self.rcf_model.boundary_detection(source_img)

        # This is hyper-parameter  we can change
        for i in rcf_inputs_ref :
            fmax = np.max(i)

            ret3, th3_rcf = cv2.threshold(i, fmax*0.5, fmax*1, cv2.THRESH_BINARY)
            th3_rcf = th3_rcf.astype('uint8')
            th3_rcf  = 255 -th3_rcf
            plt.imshow(th3_rcf, cmap='gray')
            plt.show()

        for i in rcf_inputs_src:
            fmax = np.max(i)

            ret3, th3_rcf = cv2.threshold(i, fmax * 0.5, fmax * 1, cv2.THRESH_BINARY)
            th3_rcf = th3_rcf.astype('uint8')
            th3_rcf = 255 - th3_rcf
            plt.imshow(th3_rcf, cmap='gray')
            plt.show()



        exit(0)

        w,h = th3_rcf.shape
        th3_resize = (255*cv2.resize(th3,(h,w))).astype('uint8')
        plt.imshow(th3_resize*th3_rcf, cmap='gray')
        plt.show()


        # print(rcf_inputs_src[2])
        # matchingImgMaskCorrRCF = self.MatchingMask(rcf_inputs_ref[2].astype("uint8"),
        #                                                     rcf_inputs_src[2].astype("uint8"))
        #
        # plt.imshow(matchingImgMaskCorrRCF, cmap='gray', vmin=0, vmax=1)
        # plt.show()



        exit(0)
        # # sub-two images
        # sub_img = rcf_inputs_ref[1] - rcf_inputs_src[1]
        # sub_img_abs = np.abs(sub_img)
        # sub_img_abs = 255*(sub_img_abs-np.min(sub_img_abs))/(np.max(sub_img_abs)-np.min(sub_img_abs))
        # h,w = sub_img_abs.shape
        # plt.imshow(sub_img_abs, cmap='gray')
        # plt.show()
        # sub_img_abs = 255 *( 1- ((sub_img_abs - np.min(sub_img_abs)) / (np.max(sub_img_abs) - np.min(sub_img_abs))))
        # h, w = sub_img_abs.shape
        # plt.imshow(sub_img_abs, cmap='gray')
        # plt.show()
        # sub_img_abs = sub_img_abs.astype('uint8')
        # sub_img_abs_blur = cv2.GaussianBlur(sub_img_abs, (5, 5), 0)
        # ret3, th3 = cv2.threshold(sub_img_abs_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # plt.imshow(th3, cmap='gray')
        # plt.show()
        # resize_matchingImgMaskCorr=cv2.resize(matchingImgMaskCorr,(w,h))
        # res_th3Corr = np.multiply(th3,resize_matchingImgMaskCorr)
        # plt.imshow(res_th3Corr, cmap='gray')
        # plt.show()

        # Let's  try Sharp !!
        h,w = ref_img.shape[0:2]
        print(h,w)
        if h % 2 == 0 : h = h +1
        if w % 2 == 0: w = w + 1
        print(h, w)

        ref_img_ga = cv2.GaussianBlur(ref_img,(h,w),0)
        source_img_ga= cv2.GaussianBlur(source_img,(h,w),0)

        sharp_ref_img = cv2.addWeighted(ref_img,1.5,ref_img_ga,-0.25,0)
        sharp_source_img = cv2.addWeighted(source_img, 1.5, source_img_ga, -0.25, 0)

        plt.imshow(ref_img_ga)
        plt.show()
        plt.imshow(sharp_ref_img)
        plt.show()
        plt.imshow(source_img_ga)
        plt.show()
        plt.imshow(sharp_source_img)
        plt.show()

        if (len(sharp_ref_img.shape) == 3):
            sharp_ref_img_gray = cv2.cvtColor(sharp_ref_img, cv2.COLOR_RGB2GRAY)
        else:
            sharp_ref_img_gray = sharp_ref_img.copy()

        if (len(sharp_source_img.shape) == 3):
            sharp_source_img_gray = cv2.cvtColor(sharp_source_img, cv2.COLOR_RGB2GRAY)
        else:
            sharp_source_img_gray = sharp_source_img.copy()
        matchingImgMaskCorr = self.MatchingMask(sharp_ref_img_gray, sharp_source_img_gray)
        cv2.imshow("matchingImgMaskCorr", matchingImgMaskCorr)
        cv2.waitKey(0)

        exit(0)
        rcf_inputs_ref = [final, graident,refined_img, lsd_grd] = self.rcf_model.boundary_detection(sharp_ref_img)
        rcf_inputs_src = [final, graident, refined_img, lsd_grd] = self.rcf_model.boundary_detection(sharp_source_img)

        for i in rcf_inputs_ref:
            plt.imshow(i)
            plt.show()
        for i in rcf_inputs_src:
            plt.imshow(i)
            plt.show()

        sub_img = rcf_inputs_ref[1] - rcf_inputs_src[1]
        plt.imshow(sub_img)
        plt.show()

        sub_img_abs = np.abs(sub_img)
        plt.imshow(sub_img_abs)
        plt.show()


        ######################

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        imgLaplacian = cv2.filter2D(sub_img_abs, cv2.CV_32F, kernel)
        sharp = np.float32(sub_img_abs)

        imgResult = sharp - imgLaplacian
        # convert back to 8bits gray scale
        imgResult = np.clip(imgResult, 0, 255)

        plt.imshow(imgResult)
        plt.show()


        exit(0)




        ########### Part 1 : Edge Detection + Close Gaps in Edges ############

         #### Let try edge detection using Canny with different paramters
        for blurFact in [3] :
            for minValueThresh in [100] :
                edges_srcs = cv2.Canny(cv2.GaussianBlur(source_img, (blurFact, blurFact), 0), minValueThresh, 2*minValueThresh)
                # cv2.imshow("the edge map using canny src ", edges_srcs)
                edges_ref = cv2.Canny(cv2.GaussianBlur(ref_img, (blurFact, blurFact), 0), minValueThresh, 2*minValueThresh)
                # cv2.imshow("the edge map using canny ref ", edges_ref)
                # cv2.waitKey(0)


        #Close Gaps and make continues edges maps !
        cv2.destroyAllWindows()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        edges_srcs = cv2.morphologyEx( edges_srcs, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow(" The Edges of Canny ", edges_srcs)
        # cv2.imwrite('CannyMask.png', edges_srcs)

        #
        # # cv2.waitKey(0)
        # #### Let try edge detection using sobel and normal gradient
        # gray_source_img = cv2.cvtColor(source_img,cv2.COLOR_BGR2GRAY)
        # sobelx = cv2.Sobel(gray_source_img,cv2.CV_32F,1,0,ksize=5)
        # sobely = cv2.Sobel(gray_source_img, cv2.CV_32F, 0,1, ksize=5)
        # soblex_abs = cv2.convertScaleAbs(sobelx)
        # sobley_abs = cv2.convertScaleAbs(sobely)
        # mag = cv2.addWeighted(soblex_abs,0.5,sobley_abs,0.5,0)
        # cv2.imshow("mag of graidnt", mag)
        # print(mag)
        # # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # ##### Let try the edge detetion using Line Ssegment
        # #Read gray image
        # img = gray_source_img
        # #Create default parametrization LSD
        # lsd = cv2.createLineSegmentDetector(0)
        # #Detect lines in the image
        # lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
        # #Draw detected lines in the image
        # drawn_img = lsd.drawSegments(img,lines)
        # print(lines)
        # # drawn_img = lsd.drawSegments(np.zeros(img.shape),lines)
        # #Show image
        # cv2.imshow("LSD",drawn_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        ########### Part 2 : Pattren or Template Matching ############

        numsplit = 4
        ntry = 2
        split_factor = 1

        for angle in [0]:
            ref_rotated_color = self.rotate_image(ref_img, angle)
            source_rotated_color= self.rotate_image(source_img, angle)
            heatmap, _ = self.split_img_overlapp(ref_rotated_color, source_rotated_color, numsplit ** split_factor)
            heatmap_rotated = self.rotate_image(heatmap, angle)



            # binary_heatmap = heatmap_rotated
            # binary_heatmap[binary_heatmap>0] = 1
            # print(binary_heatmap)
            # binary_heatmap = heatmap_rotated[heatmap_rotated > 0] = 1
            # cv2.imshow("binary heatmap ", binary_heatmap)


        ######## Start build more fitting border !! ##########
        for q in [1]:
            split_factor = split_factor + 1
            numberofParts= np.sqrt(numsplit**split_factor)
            r_len = int(ref_img.shape[0] // numberofParts)  # row
            c_len = int(ref_img.shape[1] // numberofParts)  # cols

            for i in range(0,int(numberofParts)):
                  for j in range(0, int(numberofParts)):

                      crop_mask = heatmap_rotated[i*r_len:i*r_len+ r_len,j*c_len:j*c_len+c_len]

                      if(int(np.sum(crop_mask))==r_len*c_len):
                          self.acc_mask[i*r_len:i*r_len+ r_len,j*c_len:j*c_len+c_len] = crop_mask
                      else:

                          mini_patch_ref = ref_img[i*r_len:i*r_len+ r_len,j*c_len:j*c_len+c_len]
                          mini_patch_src = source_img[i*r_len:i*r_len+ r_len,j*c_len:j*c_len+c_len]
                          mini_mask,_ = self.split_img_overlapp(mini_patch_ref , mini_patch_src, numsplit**split_factor,numberofParts*0.01)
                          # cv2.imshow("mini_mask ", mini_mask)
                          # cv2.imshow("crop_mask +  mini_mask ", crop_mask+mini_mask)
                          # cv2.waitKey(0)
                          # cv2.destroyAllWindows()
                          # self.acc_mask[i * r_len:i * r_len + r_len, j * c_len:j * c_len + c_len]= crop_mask+mini_mask
                          self.acc_mask[i * r_len:i * r_len + r_len,j * c_len:j * c_len + c_len] = crop_mask + mini_mask
            # cv2.imshow("  The acc each step", self.acc_mask)
            # cv2.waitKey(0)


        ############ REFINE 1 :
        # we can see that the last rows and clos is black which we need to fill with
        # number of  copy like small patch S = 4
        # s
        s= 3;
        self.acc_mask[:,-s:] =    self.acc_mask[:,self.acc_mask.shape[1]-(2*s+1):self.acc_mask.shape[1]-(s+1)]
        # cv2.imshow(" The acc after all angels fill horz ", self.acc_mask)
        # cv2.waitKey(0)
        self.acc_mask[-s:,:] = self.acc_mask[self.acc_mask.shape[0] - (2 * s + 1):self.acc_mask.shape[0] - (s + 1),:]
        # cv2.imshow(" The acc after all angels fill  vert ", self.acc_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #############################################


        ############ REFINE 2:
        # Ensure that black area all of it inside the edges to get best results !!

        mirr_mask = self.acc_mask.copy()
        mirr_mask = 1 - mirr_mask

        # close Gaps!
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mirr_mask = cv2.morphologyEx(mirr_mask, cv2.MORPH_ERODE, kernel)
        # cv2.imshow(" The acc with morph ", mirr_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # self.outVideo.write(np.concatenate((mirr_mask, edges_srcs), axis=1))

        ############ REFINE 2 ############# :
        # Using morphy logical operators to deal

        binary_heatmap =1- mirr_mask
        binary_heatmap[binary_heatmap > 0] = 1

        self.water_occ(edges_srcs,binary_heatmap)
        # cv2.imshow("binary heatmap ", binary_heatmap)
        # cv2.imwrite('binary_mask_templateMatching.png', 255*binary_heatmap ) # for i in range(0, h):
        # self.acc_mask = 1- mirr_mask

        ################ REFINE 3 ########################
        # snapping the black area
        self.water_occ(edges_srcs,self.acc_mask)
        # cv2.imshow(" Final The acc before snapping",  255-self.refine_mask_build)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        # self.occ_mask_frq,self.occ_mask_color = self.split_img_overlapp(ref_img, source_img, numsplit)
        # cv2.imshow('Final Mask Freq', self.occ_mask_frq)
        # cv2.imshow('Final Mask Color',self.occ_mask_color)

        self.MaskNonTransform = 255-self.refine_mask_build;
        # kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.warpmaskOcc = cv2.warpPerspective(self.MaskNonTransform, h_transform, (self.FrameWith, self.FrameHeight))
        # cv2.imshow("Final Mask Freq vs Modify", np.concatenate((self.MaskNonTransform,  self.warpmaskOcc), axis=1))
        # cv2.imshow("Final Mask Freq vs Modify",  self.warpmaskOcc + self.umask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.morphy_mask = cv2.morphologyEx(self.warpmaskOcc + self.umask, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("morphy_mask",  self.morphy_mask)
        # self.OutVideMasked.write(np.concatenate(( self.morphy_mask, self.warpmaskOcc), axis=1))
        # cv2.waitKey(0)
    def water_occ(self,edges_map, template_mask):

        self.template_edge_mask = 255*(template_mask)
        # cv2.imshow(" template_edge_mask", self.template_edge_mask)
        # cv2.waitKey(0)
        # White = 0 #or 255 for white and 0 black
        self.refine_mask= self.template_edge_mask.copy()

        print("sweep algorthim start")
        # self.sweep_mask(edges_map,self.template_edge_mask)
        # cv2.imshow(" refine", self.refine_mask)
        # cv2.waitKey(0)
        print("sweep algorthim End")
        print("the edges")
        # cv2.imshow(" edges", edges_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        print("start filling contour")
        self.template_edge_mask = (255-self.refine_mask).copy()
        rows, cols = np.where(  self.template_edge_mask==255)
        self.backtrack_matrix = np.zeros( self.template_edge_mask.shape , dtype=bool)
        numpoints = 4
        arr_index = np.random.choice(range(0,len(rows)), numpoints)
        print("the index random",arr_index)
        # print("the shape of template",self.template_edge_mask.shape)

        self.refine_mask_build = np.zeros( self.template_edge_mask.shape)
        print("the shape of edges", edges_map.shape)
        for i in range(0,numpoints):
            index_i,index_j = rows[arr_index[i]],cols[arr_index[i]]
            print(index_i,index_j)
            # cv2.imshow("edges_map",edges_map)
            # cv2.imshow("self.refine_mask_build", edges_map)
            # cv2.waitKey(0)
            self.fillingConturiterative(edges_map,  self.refine_mask_build, self.backtrack_matrix,index_i,index_j)
            # cv2.imshow("    self.refine_mask_build",     self.refine_mask_build)
            # cv2.waitKey(0)

            # we need to subtract those who been chocen in backtrack matrix to let the random pick them !!!!!

        # self.outVideo.release()

        return 0
    def fillingConturiterative(self,edges_map,template_match,backtrack_matrix,index_i,index_j):

        (h, w) = template_match.shape
        list_index = [(index_i,index_j)] # list of tuple of index

        curr_indexI = list_index[0][0]
        curr_indexJ = list_index[0][1]
        while(len(list_index)>0):
            curr_indexI = list_index[0][0]
            curr_indexJ = list_index[0][1]

            if (curr_indexJ >= w or curr_indexJ < 0):
                # print("(index_j >= w | index_j <= 0 )")
                del list_index[0]
                continue
            if (curr_indexI >= h or curr_indexI < 0):
                # print("(index_i >= h | index_i <= 0 )")
                del list_index[0]
                continue
            if (edges_map[curr_indexI, curr_indexJ] == 255):
                # print("edges_map[index_i,index_j]==255")
                backtrack_matrix[curr_indexI, curr_indexJ] = True
                del list_index[0]
                continue

            if backtrack_matrix[curr_indexI, curr_indexJ] == True:
                # print("self.backtrack_matrix[index_i,index_j] == True")
                del list_index[0]
                continue

            template_match[curr_indexI, curr_indexJ] = 255;
            backtrack_matrix[curr_indexI, curr_indexJ] = True

            # self.outVideo.write(np.concatenate((template_match, edges_map), axis=1))

            del list_index[0]

            list_index.insert(0,(curr_indexI, curr_indexJ-1))
            list_index.insert(0, (curr_indexI+1 , curr_indexJ))
            list_index.insert(0, (curr_indexI-1 , curr_indexJ))
            list_index.insert(0, (curr_indexI, curr_indexJ+1))

        # cv2.imshow('template_match', template_match)
        # cv2.waitKey(0)
    def fillingContur(self,edges_map,template_match,index_i,index_j):

        (h, w) = template_match.shape
        #overflow
        if(index_j >= w or index_j < 0 ) :
            # print("(index_j >= w | index_j <= 0 )")
            # cv2.imshow("  self.template_edge_mask_modifed", template_match)
            # cv2.waitKey(0)
            return template_match
        if(index_i >= h or index_i < 0 ) :
            # print("(index_i >= h | index_i <= 0 )")
            # cv2.imshow("  self.template_edge_mask_modifed", template_match)
            # cv2.waitKey(0)
            return  template_match
        if (edges_map[index_i,index_j]==255):
            print("edges_map[index_i,index_j]==255")
            self.backtrack_matrix[index_i, index_j] = True
            return template_match
        if(self.backtrack_matrix[index_i,index_j] == True):
            print("self.backtrack_matrix[index_i,index_j] == True")
            return  template_match


        print(index_i,index_j)
        template_match[index_i, index_j] =255;
        self.backtrack_matrix[index_i, index_j] = True
        # 8-n
        self.fillingContur(edges_map,template_match, index_i, index_j - 1)
        self.fillingContur(edges_map,template_match, index_i - 1, index_j)
        self.fillingContur(edges_map,template_match, index_i , index_j + 1)
        self.fillingContur(edges_map,template_match, index_i + 1, index_j)


        # return self.template_edge_mask
    def sweep_mask(self,edges_map,template_mask):


        self.refine_mask = template_mask.copy()

        # self.refine_mask_ver = binary_mask
        (h,w) = template_mask.shape

        ###################  Horizontal  Snapping right to left ##################

        for i in range(0,h):
            for j in range(0, w):
                flag = False
                if(template_mask[i,j]==0 and template_mask[i,j-1]==255):

                    #Left Direction

                    l_m=j
                    while(l_m> 0):
                          if(edges_map[i,l_m]==255):
                              break;

                          l_m= l_m-1

                    self.refine_mask[i, l_m:j] = 0
                    self.outVideo.write(np.concatenate(( self.refine_mask, edges_map), axis=1))

                    #
                    # # right Direction
                    # r_m = j
                    # while ( r_m < w):
                    #     if (edges_map[i, r_m] == 255):
                    #         # self.refine_mask_horz[i, m:j] = 0
                    #         break;
                    #     r_m = r_m + 1
                    #
                    #
                    # if(np.abs(r_m-j) <= np.abs(l_m-j)):
                    #          self.refine_mask[i, j:r_m] = 0
                    # else:
                    #     self.refine_mask[i, l_m:j] = 0

                elif(template_mask[i, j] == 0 and template_mask[i, j - 1] == 0):
                    self.refine_mask[i, j] = 0
                else:
                    self.refine_mask[i,j]=255


        # cv2.imshow("The refine_mask map hoz", (255-self.refine_mask)+edges_map)
        # cv2.imwrite('refine_mask_right_to_left.png', (255-self.refine_mask)+edges_map)
        # cv2.waitKey(0)

        ###################  Horizontal  Snapping right to left ##################
        template_mask = self.refine_mask.copy()
        for i in range(0, h):
            for j in range(0, w):
                flag = False
                if (template_mask[i, j] == 255 and template_mask[i, j - 1] == 0):

                    # right Direction
                    r_m = j
                    while ( r_m < w):
                        if (edges_map[i, r_m] == 255):
                            # self.refine_mask_horz[i, m:j] = 0
                            break;
                        r_m = r_m + 1
                    self.refine_mask[i,j:r_m] = 0
                    self.outVideo.write(np.concatenate((self.refine_mask, edges_map), axis=1))
                elif (template_mask[i, j] == 0 and template_mask[i, j - 1] == 0):
                    self.refine_mask[i, j] = 0
                else:
                    self.refine_mask[i, j] = 255

        # cv2.imshow("The refine_mask map hoz", (255 - self.refine_mask) + edges_map)
        # cv2.imwrite('refine_mask_right_to_left.png', (255 - self.refine_mask) + edges_map)
        # cv2.waitKey(0)

        ###################  Horizontal  Snapping top to bottom ##################
        template_mask = self.refine_mask.copy()

        for j in range(0, w):
            for i in range(0, h):

                if (template_mask[i, j] == 255 and template_mask[i - 1, j] == 0):

                    # Bottom Direction
                    b_m = i

                    while (b_m < h):
                        if (edges_map[b_m, j] == 255):
                            break;
                        b_m = b_m + 1

                    self.refine_mask[i-1:b_m,j] = 0
                    self.outVideo.write(np.concatenate((self.refine_mask, edges_map), axis=1))
                    # cv2.imshow("The refine_mask map vertical", (255 - self.refine_mask) + edges_map)
                    # cv2.waitKey(0)  ###########

        ###################  Horizontal  Snapping bottom to up ##################

        template_mask = self.refine_mask.copy()
        for j in range(0, w):
            for i in range(0, h):

                if (template_mask[i, j] == 0 and template_mask[i - 1, j] == 255):

                    # up Direction
                    u_m = i
                    while (u_m > 0):
                        if (edges_map[u_m, j] == 255):
                            break;
                        u_m = u_m - 1

                    self.refine_mask[u_m:i, j] = 0
                    self.outVideo.write(np.concatenate((self.refine_mask, edges_map), axis=1))
                elif (template_mask[i, j] == 0 and template_mask[i - 1, j] == 0):
                    self.refine_mask[i, j] = 0
                else:
                    self.refine_mask[i, j] = 255

        # self.refine_mask_ver = cv2.morphologyEx( self.refine_mask_ver, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("The refine_mask map vertical", (255-self.refine_mask)+edges_map)
        # cv2.imwrite('refine_mask_edges.png', (255-self.refine_mask)+edges_map)
        # cv2.imwrite('refine_mask_Noedges.png', (255 - self.refine_mask) )
        # self.outVideo.release()
        # cv2.waitKey(0)




        #
        # cv2.destroyAllWindows()
    def split_img_overlapp(self,ref_img, source_img, numsplit,threshError=0.02,flag_init=True):

        # ref_img : the reference image, the rectified
        # source_img : it may have occluded parts

        mask_frq = np.zeros((ref_img.shape[0], ref_img.shape[1]))
        mask_color = np.zeros((ref_img.shape[0], ref_img.shape[1]))
        heat_map = np.zeros((ref_img.shape[0], ref_img.shape[1]))

        # cv2.imshow('sImage', source_img)
        # print('sImage shape',source_img.shape)
        # cv2.imshow('Image', ref_img)
        # print('Image shape',ref_img.shape)
        # cv2.waitKey(0)

        ## how much split's I need

        r_len = int(ref_img.shape[0] // np.sqrt(numsplit))  # row
        c_len = int(ref_img.shape[1] // np.sqrt(numsplit))  # cols
        #        print('size of mini-image' , (r_len,c_len))
        print("the threshError =  ",threshError)
        cnt = 1
        #smallest patch ever !
        S_patch = 4
        # for i in range(0, ref_img.shape[0] - r_len + 1):
        #     for j in range(0, ref_img.shape[1] - c_len + 1):
        for i in range(0, ref_img.shape[0] ):
            for j in range(0, ref_img.shape[1]):
                cnt = cnt + 1


                if(np.abs(ref_img.shape[1]-j) < S_patch):
                    heat_map[i, j] =  heat_map[i, j-1]

                if(np.abs(ref_img.shape[0]-i) < S_patch ):
                    heat_map[i, j] = heat_map[i-1, j]

                # smallest patch ever
                if(np.abs(ref_img.shape[1]-j) >= S_patch and  np.abs(ref_img.shape[0]-i) >= S_patch ):
                    # print("The template start point  =", (i, j), ",the shape whole image = ", ref_img.shape)
                    if (j + c_len < ref_img.shape[1]) and (r_len + i < ref_img.shape[0]):
                             crop_ref_img = ref_img[i:r_len + i, j:j + c_len]
                             crop_source_img = source_img[i:r_len + i, j:j + c_len]
                             end_i = r_len + i
                             end_j = j + c_len
                             # print("Proper Template")
                    elif(j + c_len >= ref_img.shape[1] and r_len + i <= ref_img.shape[0]):
                        crop_ref_img = ref_img[i:r_len + i, j:ref_img.shape[1]]
                        crop_source_img = source_img[i:r_len + i, j:source_img.shape[1]]
                        end_i = r_len + i
                        end_j = source_img.shape[1]
                        # print("Template Cols  exceed")
                        # cv2.waitKey(0)
                    elif (j + c_len <= ref_img.shape[1] and r_len + i >= ref_img.shape[0]):
                        crop_ref_img = ref_img[i:ref_img.shape[0], j:j + c_len]
                        crop_source_img = source_img[i:source_img.shape[0], j:j + c_len]
                        # print("Template Rows  exceed")
                        end_i = ref_img.shape[0]
                        end_j = j + c_len

                        # cv2.waitKey(0)
                    else:
                        crop_ref_img = ref_img[i:ref_img.shape[0],  j:ref_img.shape[1]]
                        crop_source_img = source_img[i:source_img.shape[0],  j:source_img.shape[1]]
                        # print("Template Rows and Cols exceed")
                        end_i = ref_img.shape[0]
                        end_j = source_img.shape[1]
                        # cv2.waitKey(0)




                #
                # cv2.imshow('crop_ref_img', crop_ref_img)
                # print('crop_ref_img shape',crop_ref_img.shape)
                # cv2.imshow('crop_source_img', crop_source_img)
                # print('crop_source_img shape',crop_source_img.shape)
                # cv2.waitKey(0)

                                   ### Gray Template matching

                res = cv2.matchTemplate(crop_ref_img, crop_source_img, cv2.TM_SQDIFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                # print("The max_val is ",max_val)

                if (max_val <threshError):
                    # heat_map[i,j] = heat_map[i, j] + 1
                    heat_map[i:end_i, j:end_j] = heat_map[i:end_i, j:end_j] + 1
                    # heat_map[i:r_len + i, j:j + c_len] = heat_map[i:r_len + i, j:j + c_len] + 1



        #
        # if(flag_init==True):
        #  # cv2.imshow("The heat-map", heat_map)
        #  # cv2.waitKey(0)
        #  # print(heat_map)
        #  # plt.imshow(heat_map)
        #  # plt.show()



        return heat_map,heat_map
    def split_img(self,ref_img, source_img, numsplit):

            # ref_img : the reference image, the rectified
            # source_img : it may have occluded parts

            mask_frq = np.zeros((ref_img.shape[0], ref_img.shape[1]))
            mask_color = np.zeros((ref_img.shape[0], ref_img.shape[1]))

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
                        crop_ref_img_gray = cv2.cvtColor(crop_ref_img, cv2.COLOR_RGB2GRAY)
                    else:
                        crop_ref_img_gray = crop_ref_img
                    if (len(crop_source_img.shape) == 3):
                        crop_source_img_gray = cv2.cvtColor(crop_source_img, cv2.COLOR_RGB2GRAY)
                    else:
                        crop_source_img_gray = crop_source_img

                    ##################################
                    crop_ref_img_psr,_,self.crop_ref_imgFrequency = self.correlate(np.float32(crop_ref_img_gray),np.float32(crop_ref_img_gray))
                    crop_source_psr, _,self.crop_source_imgFrequency= self.correlate(np.float32(crop_ref_img_gray),np.float32(crop_source_img_gray))




                    print(" ratio psr of ref/source ",crop_source_psr/crop_ref_img_psr)
                    print(" crop_source_img_psr and crop_ref_psr"  ,crop_source_psr,crop_ref_img_psr)
                    #                     # cv2.imshow("the 2 ref and source",np.concatenate((crop_source_img_gray, crop_ref_img_gray), axis=1))
                    #                     # cv2.waitKey(0)
                    if(crop_source_psr ==0 and crop_ref_img_psr ==0 ):
                        print("the size of template ",crop_source_img_gray.shape)
                    # ###### Frequency of the parts
                    # if (crop_source_psr> crop_ref_img_psr  or np.abs( 0.9 * crop_ref_img_psr - crop_source_psr) < 0.2*crop_ref_img_psr):

                    if (crop_ref_img.shape[0] <= 2 or crop_ref_img.shape[1] <= 2):


                        # res = cv2.matchTemplate(crop_source_img_gray, crop_ref_img_gray, cv2.TM_SQDIFF_NORMED)
                        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        # print("the  value of matching", max_val)
                        if (crop_source_psr/crop_ref_img_psr > 0.95):
                            mask_frq[start_r:r_len + start_r, start_c:start_c + c_len] = 1
                        else:
                            mask_frq[start_r:r_len + start_r, start_c:start_c + c_len] = 0
                            # crop_mask_frq, _ = self.split_img(crop_ref_img, crop_source_img, numsplit)
                        # mask_frq[start_r:r_len + start_r, start_c:start_c + c_len] = crop_mask_frq
                    elif (crop_source_psr/crop_ref_img_psr) > 0.9:
                            mask_frq[start_r:r_len + start_r, start_c:start_c + c_len] = 1
                    else:
                        crop_mask_frq,_= self.split_img(crop_ref_img, crop_source_img, numsplit)
                        mask_frq[start_r:r_len + start_r, start_c:start_c + c_len] = crop_mask_frq




                    # ### Gray Template matching
                    #
                    # res = cv2.matchTemplate(crop_source_img_gray, crop_ref_img_gray, cv2.TM_CCORR_NORMED)
                    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    #
                    # # concate_img = np.concatenate((crop_source_img_gray, crop_ref_img_gray),axis=1)
                    # # concate_img =  cv2.resize(concate_img,(480,360))
                    # # cv2.imshow("the 2 ref and source",concate_img)
                    # # print("Value Of Matchings ", max_val)
                    # # cv2.waitKey(0)
                    #
                    #
                    # # if (crop_source_psr/crop_ref_img_psr > 0.8 or max_val> 0.99 ):
                    # if ( max_val > 0.99):
                    #      mask_color[start_r:r_len + start_r, start_c:start_c + c_len] = 1
                    #
                    # elif (crop_ref_img.shape[0] <= 4 or crop_ref_img.shape[1] <= 4):
                    #     mask_color[start_r:r_len + start_r, start_c:start_c + c_len] = 0
                    # else:
                    #     _, crop_mask_color = self.split_img(crop_ref_img, crop_source_img, numsplit)
                    #     mask_color[start_r:r_len + start_r, start_c:start_c + c_len] = crop_mask_color
                    #     # mask_color[start_r:r_len + start_r, start_c:start_c + c_len] = 0


            return mask_frq,mask_color


