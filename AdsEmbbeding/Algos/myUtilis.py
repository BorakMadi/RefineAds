# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:47:00 2019

@author: Borak_Madi
"""
import cv2
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	# if the left mouse button was clickeds
	if event == cv2.EVENT_LBUTTONDOWN:
         print(param)
         param.append((x, y))
#         cv2.circle(param,(x,y),3,(0,0,255),-1)
#         cv2.imshow('image',param)
        
        
def points4Polygon(img,polygon_pts):
  
   cv2.namedWindow('image')
   cv2.setMouseCallback('image',click_and_crop, polygon_pts)
  # keep looping until the 'q' key is pressed
  
   clone = img.copy()
   while True:
	# display the image and wait for a keypress
      for pt in polygon_pts:
           cv2.circle(img,pt,3,(0,0,255),-1)
      cv2.imshow("image", img)
      key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
      if key == ord("r"):
          img = clone.copy()
 
	# if the 'c' key is pressed, break from the loo
      if(len(polygon_pts)==4):
		     break
   
   cv2.destroyAllWindows()
   
 
    
if __name__ == '__main__':
    print("Hellow World")
    
    polygon_pts= []
    
    frame = cv2.imread('futurama-2.jpg')
    points4Polygon(frame,polygon_pts)
    print(polygon_pts)
  
