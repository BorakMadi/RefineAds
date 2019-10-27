# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 18:18:49 2019

@author: Borak_Madi
"""

import cv2 as cv2
import numpy as np
binary_match = True# Hamming-Distance or False= Kd-Tree ,FLANN


#convert bbox to 4 points --> [x y w h ]  to 4 points
def bbox_4points(b):
   rect_point =  (b[0],b[1]),(b[0] + b[2],b[1]),(b[0] + b[2],b[1] + b[3]),(b[0],b[1]+b[3])
   return rect_point

def  setup(detector_type,descriptor_type):
    binary_match = True
    # Detector !      
    if detector_type == 'FAST':
            detector = cv2.FastFeatureDetector_create()
    if detector_type == 'ORB':
            detector =  cv2.ORB_create(nfeatures=100000)
    if detector_type == 'AGAST':
            detector = cv2.AgastFeatureDetector_create()    
    if detector_type == 'SURF':
           detector = cv2.xfeatures2d.SURF_create()
    if detector_type == 'SIFT':
           detector = cv2.xfeatures2d.SIFT_create()

    if detector_type == 'AKAZE':
        detector = cv2.AKAZE_create()

    #Descriptor
    if descriptor_type == 'FAST':
               descriptor = cv2.BRISK_create()
    if descriptor_type == 'BRIEF':
                descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()                  
    if descriptor_type == 'BRISK':
                descriptor = cv2.BRISK_create()
    if descriptor_type == 'ORB':
               descriptor = cv2.ORB_create() 
    if descriptor_type == 'FREAK':
               descriptor = cv2.xfeatures2d.FREAK_create()              
    
    if descriptor_type == 'AKAZE':
        descriptor = cv2.AKAZE_create()

    if descriptor_type == 'SIFT':
           descriptor = cv2.xfeatures2d.SIFT_create()
           binary_match = False


    if descriptor_type == 'SURF':
           descriptor = cv2.xfeatures2d.SURF_create()
           binary_match = False
        

    if binary_match is False :
        
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            match_type = "FLANN Matcher"
#            matches = flann.knnMatch(self.features_descriptors[-1], descriptor, k=2)
    else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            # matcher  = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
            match_type = "Hamming Distance"
#            matches = bf.knnMatch(self.features_descriptors[-1], descriptor, k=2)
            
            
            

    print('Detector = ', detector_type)
    print('Desc = ', descriptor_type)
    print('Match =',match_type)
    
    return detector,descriptor,matcher


class FeatureExtraction:
    """
    Object that detect and extract feature points and match "
    """ 
    def __init__(self, detector='ORB',descriptor='ORB'):
        
        # Methods 
        self.detMeth ,self.desMeth,self.matcher = setup(detector,descriptor)
        self.des_feats = []
        self.kp_feats = []
        self.frameWKeyPoints = []
    # === init_frame ===
    def init_frame(self,frame,mask_pts):
        
    # we need to extract feature ony from the desired region! , :D for simplicity we extract from box     
    # rect_points = bbox_4points(bbox)

        mask_rect = np.zeros((frame.shape[0:2]),dtype=np.uint8) 
        xys = np.array(mask_pts,np.int32)
        cv2.fillConvexPoly(mask_rect,xys, 255)
        self.Extmask = mask_rect
        self.frameWKeyPoints,self.kp_feats,self.des_feats = self.detect_and_compute(frame, self.Extmask)

              
    def detect_and_compute(self,frame,mask=None):
    
         """Detect and compute interest points and their descriptors."""
         if(len(frame.shape) ==3):
             frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
         kp  =  self.desMeth.detect(frameGray,mask)
         kp, des= self.desMeth.compute(frameGray, kp)

         return self.drawKeypoints(frame,mask), kp, des
    
  
    def matchFeatures(self, descriptor ,k=2):
        # we need to compute the warp mask :D ,right now we extract from all ! 
#        dummy, kp, des = self.detect_and_compute(frame,self.detMeth,self.desMeth)

        if binary_match is True:
               matches = self.matcher.knnMatch(self.des_feats, descriptor, k)
        else:
            matches = self.mat

        return matches
        
    def drawKeypoints(self,frame,mask=None):
        # utility function to draw keypoints on image

        kp = self.desMeth.detect(frame, mask)
        kp, des = self.desMeth.compute(frame, kp)
        for mark in kp:
               frame= cv2.drawMarker(frame, tuple(int(i) for i in mark.pt), color=(0, 255, 0))
        return frame


    # we need to update the mask we extract from it will give us better matching
    def updateMask(self,h):

        return  cv2.warpPerspective(self.Extmask,h,(self.Extmask.shape[1],self.Extmask.shape[0]))
        # self.Extmask = cv2.warpPerspective(self.Extmask,h,(self.Extmask.shape[1],self.Extmask.shape[0]))
           
       

if __name__ == '__main__':
        
    print('Hi World')
    
    frame = cv2.imread('futurama-2.jpg')
    cv2.waitKey(0)
    fe= FeatureExtraction('ORB','ORB' )

  # bbox is simple form of polygon , we can use 4points polygon  :D     
    bbox  =  cv2.selectROI(frame, False)
    bbox_pts = bbox_4points(bbox)
    fe.init_frame(frame,bbox_pts)
    img_kp = fe.drawKeypoints(frame,fe.kp_feats)
    cv2.imshow('KeyPoint',img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()