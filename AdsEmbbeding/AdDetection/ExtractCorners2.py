
import os, sys
import numpy as np
from PIL import Image
import cv2 as cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn import linear_model
from math import atan2, cos, sin, sqrt, pi,degrees
from numpy.linalg import lstsq
from numpy import linalg as LA
from itertools import product
import math
import scipy
from scipy import ndimage



sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from RCF_Utility import RCF_BoundaryDetector as RCFBD

from distanceMapTest import DistanceMap

import datetime

now = datetime.datetime.now()

SHOW_RESULTS = False
SHOW_COMP_RESULTS = False
refPt = []
num_p_sampling = 5
corners_4polygon = []

import enum
class Direction(enum.Enum):
    Left = 1
    Right = 2
    Down = 3
    Up = 4



# def main():
#     # img = cv2.imread('RCF_Utility/0.jpg')
#     # extract_corners2(img)
#     #print("

def pick_pionts(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        print(refPt)

#method =1 LS
#method =2 RANSAC
def snap_nearestEdge(points, edgeMap, snapSide,EdgeBoundries,corners):

    # print("pointing Snapping")
    snap_edge_points=[]
    edgeMap_tmp = edgeMap.copy()
    for i in points:
        i_x = int(i[0])
        i_y = int(i[1])
        # print(i_x,i_y)
        # cv2.imshow("The results", edgeMap_tmp)
        # cv2.waitKey(0)
        while (edgeMap_tmp[i_x, i_y] != 255):  ## it's not edge continue to snap
            edgeMap_tmp[i_x, i_y] = 255
            if (snapSide == Direction.Left):
                i_y = i_y - 1  # we take simple case !!
            if (snapSide == Direction.Right):
                i_y = i_y + 1
            if (snapSide == Direction.Up):
                i_x = i_x - 1
            if (snapSide == Direction.Down):
                i_x = i_x + 1
        snap_edge_points.append((i_x, i_y))
        # cv2.imshow("The results",edgeMap_tmp)
        # cv2.waitKey(0)

    if snapSide == Direction.Down or snapSide == Direction.Up :
        # print("Snapping Ver")
        # Add outlier data
        X,y = zip(*snap_edge_points)
        X = np.array(X).reshape(-1,1)
        y = np.array(y).reshape(-1,1)

        # Fit line using all data
        lr = linear_model.LinearRegression()
        lr.fit(y, X)
        line_X = np.arange(y.min()-5, y.max()+5)[:, np.newaxis]
        line_y = lr.predict(line_X)

        # Fit RANSAC model

        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor(residual_threshold=20)
        # ransac = linear_model.RANSACRegressor(residual_threshold=20)
        ransac.fit(y, X)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_y_ransac = ransac.predict(line_X)

        interLines(line_X, line_y_ransac, EdgeBoundries,corners)

        if SHOW_RESULTS:
            plt.imshow(edgeMap_tmp)
            lw = 2
            plt.plot(line_X,line_y, color='red', linewidth=lw, label='Linear regressor')
            plt.plot(line_X,line_y_ransac, color='cornflowerblue', linewidth=lw,
                     label='RANSAC regressor')
            plt.show()


    else:
        # print("Snapping Horz")
        # Add outlier data
        y, X = zip(*snap_edge_points)
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)

        # Fit line using all data
        lr = linear_model.LinearRegression()
        lr.fit(y, X)
        line_X = np.arange(y.min()-5, y.max()+5)[:, np.newaxis]
        line_y = lr.predict(line_X)

        # Robustly fit linear model with RANSAC algorithm
        ransac =  linear_model.RANSACRegressor(residual_threshold=20)
        ransac.fit(y, X)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_y_ransac = ransac.predict(line_X)

        interLines(line_y_ransac, line_X, EdgeBoundries,corners)
        if SHOW_RESULTS:
            plt.imshow(edgeMap_tmp)
            lw = 2
            plt.plot(line_y, line_X, color='red', linewidth=lw, label='Linear regressor')
            plt.plot(line_y_ransac,line_X, color='cornflowerblue', linewidth=lw,
                     label='RANSAC regressor')
            plt.show()

def interLines(X,y,EdgesMapRansc,corners):

    for i_x,i_y in zip(X.flatten(),y.flatten()):
        # print(EdgesMapRansc[int(i_x),int(i_y)])
        if int(EdgesMapRansc[int(i_y),int(i_x)])==255:
            corners.append((int(i_x),int(i_y)))
        else:
            cv2.rectangle(EdgesMapRansc,(int(i_x),int(i_y)),(int(i_x),int(i_y)),255,1)
            # print(EdgesMapRansc[int(i_y),int(i_x)])
            # cv2.imshow("Sda",EdgesMapRansc)
            # cv2.waitKey(0)

    # print(EdgesMapRansc)

def snap_nearestEdgesMatch(ptsLine,edgeMap, snapSide,method=1):
    # print(" --- snap_nearestEdgesMatch ---")
    edgeMap_tmp = edgeMap.copy()
    for i in ptsLine:
        i_x = i[1]
        i_y = i[0]
        cv2.imshow("The results", edgeMap_tmp)
        cv2.waitKey(0)
        while (edgeMap_tmp[i_x, i_y] != 255):  ## it's not edge continue to snap
            edgeMap_tmp[i_x, i_y] = 255
            if (snapSide == Direction.Left):
                i_y = i_y - 1  # we take simple case !!
            if (snapSide == Direction.Right):
                i_y = i_y + 1
            if (snapSide == Direction.Up):
                i_x = i_x - 1
            if (snapSide == Direction.Down):
                i_x = i_x + 1
        snap_edge_points.append((i_x, i_y))

def snap_inOut(edgeMap , mask):

    print("Not imple algorthims")



####################################### New Updated
def find_corners(pts):
    # print(pts.reshape(-1,2))
    new_array = pts.reshape(-1,2)
    x_arr = pts.reshape(-1, 2)[:,0]
    y_arr = pts.reshape(-1, 2)[:, 1]

    x_max,x_min = np.max(x_arr),np.min(x_arr)
    y_max, y_min = np.max(y_arr), np.min(y_arr)
    # print( x_max,x_min)
    # print( y_max, y_min)

    # draw 4 points
    list3 = [(a,b) for a, b in product([x_max,x_min], [y_max, y_min])]
    return list3,x_max-x_min,y_max-y_min

def calc_approx_edge(img,deg,cntr=None):
    # WE HAVE ONLY ONE MASK !.!! THIS WILL WORK

    unrot = rotateImage(img,90-deg,cntr)
    # cv2.imshow("unrot",unrot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img.shape)
    if (len(unrot.shape)>2):
        gray = cv2.cvtColor(unrot, cv2.COLOR_BGR2GRAY)
    else:
        gray = unrot.copy()
    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c);
        # Ignore contours that are too small or too large
        # if area < 1e2 or 1e5 < area:
        #     continue

        # print(c.shape)

        rot90 = unrot.copy()
        corners,x_len,y_len = find_corners(c)
        # for x in corners:
        #     cv2.circle(rot90, x, 3, (255, 255, 255), 2)
        # cv2.imshow("unrot", rot90)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        break

    return unrot,x_len,y_len,90-deg

def rotateImage(image, angle,cntr=None):

  if cntr is None:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
  else:
      image_center = cntr
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors = cv2.PCACompute(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    # cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    print(eigenvectors)

    # unrotate to get good image :D and calc the initial guess for better and easy calculation
    d1_deg = np.round(degrees(atan2(eigenvectors[0, 0], eigenvectors[0, 1])))
    d2_deg = np.round(degrees(atan2(eigenvectors[1, 0], eigenvectors[1, 1])))

    _, x_len, y_len, _ = calc_approx_edge(img, d1_deg)
    p1 = (
    int(np.round(cntr[0] + eigenvectors[0, 0] * x_len / 2)), int(np.round(cntr[1] + eigenvectors[0, 1] * x_len / 2)))
    p2 = (
    int(np.round(cntr[0] - eigenvectors[0, 0] * x_len / 2)), int(np.round(cntr[1] - eigenvectors[0, 1] * x_len / 2)))
    p3 = (
    int(np.round(cntr[0] - eigenvectors[1, 0] * y_len / 2)), int(np.round(cntr[1] - eigenvectors[1, 1] * y_len / 2)))
    p4 = (
    int(np.round(cntr[0] + eigenvectors[1, 0] * y_len / 2)), int(np.round(cntr[1] + eigenvectors[1, 1] * y_len / 2)))

    print(y_len,x_len)

    print([p1, p2, p3, p4])
    img_plot = img.copy()
    cv2.line(img_plot, p2, p1, (255,0,0),2)
    cv2.line(img_plot, p3, p4, (255,0,0),2)
    plt.figure()
    plt.imshow(img_plot)
    plt.show()
    # cv2.imshow("The res", img_plot)
    # cv2.waitKey(0)

    return _, cntr, x_len, y_len, [p1, p2, p3, p4], eigenvectors, [d1_deg, d2_deg]

def PCA_FindEdgesOrien(img):

    imageTo90, centr, x_len, y_len = 0, 0, 0, 0
    list_point = []
    # Convert image to binary
    ########### This Part is to know orientation and rotate to 90
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        print("Finding Contour ")
        area = cv2.contourArea(c);
        print(area)
        # Ignore contours that are too small or too large
        # if area < 1e2 or 1e5 < area:
        #     continue
            # Draw each contour only for visualisation purposes
            # cv2.drawContours(rotated_src, contours, i, (0, 0, 255), 2);
            # Find the orientation of each shape
        print("getOrientation ")
        _, centr, x_len, y_len, list_point, eg_vectors, new_degs = getOrientation(c, img)
        # print(c.shape)
        break

    return _, centr, x_len, y_len, list_point, eg_vectors, new_degs

import enum
class Direction(enum.Enum):
    Left = 1
    Right = 2
    Down = 3
    Up = 4

def SnapLine(edge_map,direction,len_snp,x_cor,y_cor,img_res=None):
    ############## Have to be a lot of optimzation steps for picking the right line !!! ###
    # We take the max ! right for simplicity

    score_map = np.zeros(edge_map.shape)
    rowStartEnd = y_cor.astype(int)
    colStartEnd = x_cor.astype(int)
    len_snp = len_snp // 2
    value_max, value_min = 0, np.inf
    p1, p2 = [], []


    if(direction==Direction.Left):
        for i in range(0,len_snp):

            if (value_max <= np.sum(edge_map[rowStartEnd, colStartEnd[0]])):
                p1 = (rowStartEnd, colStartEnd[0])
                value_max = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd, colStartEnd[0])
                value_min = np.sum(edge_map[rowStartEnd, colStartEnd[0]])

            score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]]= np.sum(edge_map[rowStartEnd, colStartEnd[0]])
            colStartEnd = colStartEnd -1

    if (direction == Direction.Right):
        for i in range(0, len_snp):

            if (value_max <= np.sum(edge_map[rowStartEnd, colStartEnd[0]])):
                p1 = (rowStartEnd, colStartEnd[0])
                value_max = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd, colStartEnd[0])
                value_min = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
            score_map[rowStartEnd[len(rowStartEnd) // 2], colStartEnd[0]] = np.sum(
                edge_map[rowStartEnd, colStartEnd[0]])
            colStartEnd = colStartEnd + 1

    if (direction == Direction.Up):
        for i in range(0, len_snp):
            if (value_max <= np.sum(edge_map[rowStartEnd[0], colStartEnd])):
                p1 = (rowStartEnd[0], colStartEnd)
                value_max = np.sum(edge_map[rowStartEnd[0], colStartEnd])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd[0], colStartEnd)
                value_min = np.sum(edge_map[rowStartEnd[0], colStartEnd])
            score_map[rowStartEnd[0], colStartEnd[len(colStartEnd) // 2]] = np.sum(
                edge_map[rowStartEnd[0], colStartEnd])
            rowStartEnd = rowStartEnd -1

    if (direction == Direction.Down):
        edge_map_plt = edge_map.copy()
        for i in range(0, len_snp):

            if (value_max <= np.sum(edge_map[rowStartEnd[0], colStartEnd])):
                p1 = (rowStartEnd[0], colStartEnd)
                value_max = np.sum(edge_map[rowStartEnd[0], colStartEnd])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd[0], colStartEnd)
                value_min = np.sum(edge_map[rowStartEnd[0], colStartEnd])
            score_map[rowStartEnd[0], colStartEnd[len(colStartEnd) // 2]] = np.sum(
                edge_map[rowStartEnd[0], colStartEnd])
            rowStartEnd = rowStartEnd + 1
    # Get the indices of maximum element in numpy array
    if (img_res is not None):
        if (direction == Direction.Right or direction == Direction.Left):
            delta = 10
            cv2.line(img_res, (p1[1], p1[0][0] - delta), (p1[1], p1[0][-1] + delta), 255, 3)
        else:
            delta = 10
            cv2.line(img_res, (p1[1][0] - delta, p1[0]), (p1[1][-1] + delta, p1[0]), (255, 0, 0), 3)

        if SHOW_RESULTS is True:
            fig = plt.figure()
            plt.imshow(img_res)
            fig.savefig('DrawLine' + str(direction) + '.png', dpi=fig.dpi)
            plt.show()

    return p1,p2

def findCornerByLinesV2( img,mask,contour):

    src = mask.copy()
    src = 255 * src.astype(np.uint8)
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    # Check if image is loaded successfully
    if src is None :
        print("We have Error Image-loading")
        exit(0)

    # PART 1
    ##############################################
    # ROTATE and refine to get nearest axis alignment !!!
    # Easy Snap and check !!
    ###################################################

    imageTo90, cntr, x_len, y_len, dgr1 = PCA_FindEdgesOrien(src)
    imageTo90, cntr, x_len, y_len, dgr2 = PCA_FindEdgesOrien(imageTo90)

    p1 = (int(np.round(cntr[0] + x_len / 2)), int(np.round(cntr[1] + 0 * x_len / 2)))
    p3 = (int(np.round(cntr[0] - 1 * x_len / 2)), int(np.round(cntr[1] - 0 * x_len / 2)))
    p2 = (int(np.round(cntr[0] - 0 * y_len / 2)), int(np.round(cntr[1] - 1 * y_len / 2)))
    p4 = (int(np.round(cntr[0] + 0 * y_len / 2)), int(np.round(cntr[1] + 1 * y_len / 2)))



    # First Try we will try Canny Edges of the contour ( fusion of RCF )

    refine_edges = contour.astype(np.uint8)
    refine_edges=rotateImage(refine_edges,90-dgr1,cntr)
    refine_edges = cv2.cvtColor(refine_edges, cv2.COLOR_GRAY2RGB)

    if SHOW_RESULTS is True:
        fig = plt.figure()
        plt.imshow(src)
        plt.show()
        fig = plt.figure()
        plt.imshow(imageTo90)
        plt.show()
        fig = plt.figure()
        plt.imshow(refine_edges)
        plt.show()

    # PART 2
    ##################################
    # Sharp  the edges
    ##################################

    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # so the possible negative number will be truncated
    imgLaplacian = cv2.filter2D(refine_edges, cv2.CV_32F, kernel)
    sharp = np.float32(refine_edges)
    imgResult = sharp - imgLaplacian
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)




    # PART 3
    ##################################
    # Find Proper Edges and Lines
    ##################################
        #
    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if SHOW_RESULTS is True:
        fig = plt.figure()
        plt.imshow(imgLaplacian)
        plt.show()
        fig = plt.figure()
        plt.imshow(imgResult)
        plt.show()

        fig = plt.figure()
        plt.imshow(bw)
        plt.show()

    newX, newY = src.shape[1], src.shape[0]
    bw = cv2.resize(bw, (int(newX), int(newY)))
    edge_map_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)


    # ####################################
    #  Left and Right
    ######################################
    list_points= [p1,p2,p3,p4]
    a = np.array(list_points)
    ind = np.lexsort((a[:, 0], a[:, 1]))
    pstart = (a[ind][0][0], a[ind][0][1])
    pend = (a[ind][-1][0], a[ind][-1][1])
    points = [pstart, pend]
    x_coords, y_coords = zip(*points)

    if SHOW_RESULTS is True:
       img_plot = imageTo90.copy()
       cv2.line(img_plot, points[0], points[1], 255)
       fig = plt.figure()
       plt.imshow(img_plot)
       plt.show()

    if points[0][0] == points[1][0]:  # which mean the slope undefined !
        y_coordinates = np.arange(y_coords[0], y_coords[1] + 1)
        x_coordinates = points[0][0] * np.ones(len(y_coordinates))
    else:
        x_coords = np.sort(x_coords)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        # we can do snapping along the predeincuar !
        # print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
        x_coordinates = np.arange(x_coords[0], x_coords[1] + 1)
        y_coordinates = (m * x_coordinates + c).astype(int)

    Left_p1, _ = SnapLine(bw, Direction.Left, x_len, x_coordinates, y_coordinates, edge_map_rgb)
    Right_p1, _ = SnapLine(bw, Direction.Right, x_len, x_coordinates, y_coordinates, edge_map_rgb)

    ################################################################

    # ####################################
    #  UP and Down
    ######################################
    a = np.array(list_points)
    ind = np.lexsort((a[:, 1], a[:, 0]))

    pstart = (a[ind][0][0], a[ind][0][1])
    pend = (a[ind][-1][0], a[ind][-1][1])
    points = [pstart, pend]
    x_coords, y_coords = zip(*points)

    if SHOW_RESULTS is True:
        img_plot = imageTo90.copy()
        cv2.line(img_plot, points[0], points[1], 255)
        fig = plt.figure()
        plt.imshow(img_plot)
        plt.show()

    if points[0][1] == points[0][0]:  # which mean the slope undefined !
        y_coordinates = np.arange(y_coords[0], y_coords[1] + 1)
        x_coordinates = points[0][0] * np.ones(len(y_coordinates))

    else:

        x_coords = np.sort(x_coords)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        # print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
        x_coordinates = np.arange(x_coords[0], x_coords[1] + 1)
        y_coordinates = (m * x_coordinates + c).astype(int)
        # print(x_coordinates)
        # print(y_coordinates)

    Up_p1,_ = SnapLine(bw, Direction.Up, y_len, x_coordinates, y_coordinates, edge_map_rgb)
    Down_p1, _ = SnapLine(bw, Direction.Down, y_len, x_coordinates, y_coordinates, edge_map_rgb)



    #########################################################################

    p1_left = (Left_p1[0][0], Left_p1[1])
    p2_left = (Left_p1[0][-1], Left_p1[1])

    p1_leftxy = (Left_p1[1], Left_p1[0][0])
    p2_leftxy = (Left_p1[1], Left_p1[0][-1])

    p1_Up = (Up_p1[0], Up_p1[1][0])
    p2_Up = (Up_p1[0], Up_p1[1][-1])

    p1_Upxy = (Up_p1[1][0], Up_p1[0])
    p2_Upxy = (Up_p1[1][-1], Up_p1[0])

    p1_right = (Right_p1[0][0], Right_p1[1])
    p2_right = (Right_p1[0][-1], Right_p1[1])

    p1_rightxy = (Right_p1[1], Right_p1[0][0])
    p2_rightxy = (Right_p1[1], Right_p1[0][-1])

    p1_Down = (Down_p1[0], Down_p1[1][0])
    p2_Down = (Down_p1[0], Down_p1[1][-1])

    p1_Downxy = (Down_p1[1][0], Down_p1[0])
    p2_Downxy = (Down_p1[1][-1], Down_p1[0])

    p_interLeftUp = intersection2Lines(p1_leftxy, p2_leftxy, p1_Upxy, p2_Upxy)
    # print(p_interLeftUp)

    p_interRightUp = intersection2Lines(p1_rightxy, p2_rightxy, p1_Upxy, p2_Upxy)
    # print(p_interRightUp)

    p_interLeftDown = intersection2Lines(p1_rightxy, p2_rightxy, p1_Downxy, p2_Downxy)
    # print(p_interLeftDown)
    p_interRightDown = intersection2Lines(p1_leftxy, p2_leftxy, p1_Downxy, p2_Downxy)
    # print(p_interRightDown)

    img_plot_sc = cv2.cvtColor(bw.copy(),cv2.COLOR_GRAY2RGB)
    for i in [p_interLeftUp, p_interRightUp, p_interLeftDown, p_interRightDown]:
        cv2.circle(img_plot_sc, (int(i[0]), int(i[1])), 3, (255, 0, 0), 2)

    if SHOW_RESULTS is True:

        fig = plt.figure()
        plt.imshow(img_plot_sc)
        plt.show()

    corners = (np.array(np.array([p_interLeftUp, p_interRightUp, p_interLeftDown, p_interRightDown])))
    corners = corners.reshape(4, 2, 1)
    return corners
def findCornerByLines(img,mask,contour):

    src = mask.copy()
    # print(src.dtype)
    # print(src.shape)
    src = 255*src.astype(np.uint8)
    src = cv2.cvtColor(src,cv2.COLOR_GRAY2RGB)
    # print(src.dtype)
    # print(src.shape)





    # Check if image is loaded successfully
    # Convert image to grayscale
    ###############################################################
    # Maybe be in all shape with all the rotation it's have small bug !!
    # We need to rotate to 90 also the results of RCF !!!! "D

    rotated_src = rotateImage(src, 30)
    # cv2.imshow('rotated_src', rotated_src)
    # cv2.waitKey(0)
    imageTo90, cntr, x_len, y_len,dgr1 = PCA_FindEdgesOrien(rotated_src)
    cv2.imshow('imageTo90', imageTo90)
    cv2.waitKey(0)

    imageTo90, cntr, x_len, y_len,dgr2 = PCA_FindEdgesOrien(imageTo90)

    p1 = (int(np.round(cntr[0] + x_len / 2)), int(np.round(cntr[1] + 0 * x_len / 2)))
    p3 = (int(np.round(cntr[0] - 1 * x_len / 2)), int(np.round(cntr[1] - 0 * x_len / 2)))
    p2 = (int(np.round(cntr[0] - 0 * y_len / 2)), int(np.round(cntr[1] - 1 * y_len / 2)))
    p4 = (int(np.round(cntr[0] + 0 * y_len / 2)), int(np.round(cntr[1] + 1 * y_len / 2)))

    # Lets take the results

    refine_edges = contour.astype(np.uint8)


    refine_edges =  cv2.cvtColor(refine_edges,cv2.COLOR_GRAY2RGB)
    # print(refine_edges.dtype,refine_edges.shape)


    if SHOW_RESULTS is True:
        fig = plt.figure()
        plt.imshow(rotated_src)
        plt.show()
        fig = plt.figure()
        plt.imshow(imageTo90)
        plt.show()

        fig = plt.figure()
        plt.imshow(refine_edges)
        plt.show()






    #######################
    #
    #  Loading Images and  "Try" to zero the background if it's white.!
    #
    #################
    # cv2.imshow('Source Image', refine_edges)
    # print(refine_edges.shape)
    # src[np.all(refine_edges == 255, axis=2)] = 0
    # # Show output image
    # cv2.imshow('Black Background Image', refine_edges)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #######################
    #
    #  Second Stage : we will sharpen our image in order to acute the edges
    #                   by using laplacian filter.
    #
    #################
    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # so the possible negative number will be truncated
    imgLaplacian = cv2.filter2D(refine_edges, cv2.CV_32F, kernel)
    sharp = np.float32(refine_edges)
    # print(imgLaplacian)

    imgResult = sharp - imgLaplacian
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    if SHOW_RESULTS is True:
        fig = plt.figure()
        plt.imshow(imgLaplacian)
        plt.show()
        fig = plt.figure()
        plt.imshow(imgResult)
        plt.show()


    ########################################################################################

    #######################
    #
    #  Third Stage : we will sharpen calc distance Map
    #
    #################

    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    newX, newY = src.shape[1], src.shape[0]
    bw = cv2.resize(bw, (int(newX), int(newY)))
    # cv2.imshow('black-White Image', bw)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    ####################################################################################

    edge_map_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    # Left and Right Snapping for  p2 and p4
    # We will need to look for x in axis world , or "Cols" in matrix world !!!!!!

    points = [p2, p4]
    x_coords, y_coords = zip(*points)

    # print("The lines with  ", points)
    img_plot = imageTo90.copy()
    # cv2.line(img_plot, points[0], points[1], 255)
    # cv2.imshow("SnappingLine", img_plot)
    # cv2.waitKey()

    if points[0][0] == points[1][0]:  # which mean the slope undefined !
        y_coordinates = np.arange(y_coords[0], y_coords[1] + 1)
        x_coordinates = points[0][0] * np.ones(len(y_coordinates))

    else:
        x_coords = np.sort(x_coords)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        # we can do snapping along the predeincuar !
        # print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
        x_coordinates = np.arange(x_coords[0], x_coords[1] + 1)
        y_coordinates = (m * x_coordinates + c).astype(int)

    Left_p1, _ = SnapLine(bw, Direction.Left, x_len, x_coordinates, y_coordinates, edge_map_rgb)
    Right_p1,_ = SnapLine(bw, Direction.Right, x_len, x_coordinates, y_coordinates, edge_map_rgb)

    ################################################################

    points = [p1, p3]
    x_coords, y_coords = zip(*points)
    # print("The lines with  ", points)
    img_plot = imageTo90.copy()
    # cv2.line(img_plot, points[0], points[1], 255)
    # cv2.imshow("SnappingLine", img_plot)
    # cv2.waitKey()

    if points[0][0] == points[1][0]:  # which mean the slope undefined !
        y_coordinates = np.arange(y_coords[0], y_coords[1] + 1)
        x_coordinates = points[0][0] * np.ones(len(y_coordinates))

    else:
        x_coords = np.sort(x_coords)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        # print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
        x_coordinates = np.arange(x_coords[0], x_coords[1] + 1)
        y_coordinates = (m * x_coordinates + c).astype(int)

    Up_p1,_ = SnapLine(bw, Direction.Up, y_len, x_coordinates, y_coordinates, edge_map_rgb)
    Down_p1,_ = SnapLine(bw, Direction.Down, y_len, x_coordinates, y_coordinates, edge_map_rgb)
    #########################################################################

    p1_left = (Left_p1[0][0], Left_p1[1])
    p2_left = (Left_p1[0][-1], Left_p1[1])

    p1_leftxy = (Left_p1[1], Left_p1[0][0])
    p2_leftxy = (Left_p1[1], Left_p1[0][-1])

    p1_Up = (Up_p1[0], Up_p1[1][0])
    p2_Up = (Up_p1[0], Up_p1[1][-1])

    p1_Upxy = (Up_p1[1][0], Up_p1[0])
    p2_Upxy = (Up_p1[1][-1], Up_p1[0])

    p1_right = (Right_p1[0][0], Right_p1[1])
    p2_right = (Right_p1[0][-1], Right_p1[1])

    p1_rightxy = (Right_p1[1], Right_p1[0][0])
    p2_rightxy = (Right_p1[1], Right_p1[0][-1])

    p1_Down = (Down_p1[0], Down_p1[1][0])
    p2_Down = (Down_p1[0], Down_p1[1][-1])

    p1_Downxy = (Down_p1[1][0], Down_p1[0])
    p2_Downxy = (Down_p1[1][-1], Down_p1[0])


    p_interLeftUp = intersection2Lines(p1_leftxy, p2_leftxy, p1_Upxy, p2_Upxy)
    print(p_interLeftUp)

    p_interRightUp = intersection2Lines(p1_rightxy, p2_rightxy, p1_Upxy, p2_Upxy)
    print(p_interRightUp)

    p_interLeftDown = intersection2Lines(p1_rightxy, p2_rightxy, p1_Downxy, p2_Downxy)
    print(p_interLeftDown)
    p_interRightDown = intersection2Lines(p1_leftxy, p2_leftxy, p1_Downxy, p2_Downxy)
    print(p_interRightDown)

    # img_plot_sc = cv2.cvtColor(bw.copy(),cv2.COLOR_GRAY2RGB)
    # for i in [p_interLeftUp, p_interRightUp, p_interLeftDown, p_interRightDown]:
    #     cv2.circle(img_plot_sc, (int(i[0]), int(i[1])), 3, (255, 0, 0), 2)
    #     cv2.imshow("The points", img_plot_sc)
    #     cv2.waitKey(0)

    corners = (np.array(np.array([p_interLeftUp, p_interRightUp, p_interLeftDown, p_interRightDown])))
    corners = corners.reshape(4, 2, 1)
    return corners

def Score_line(edge_map, normal,pstart_rot,pend_rot,zlen, img_gray=None):



    zlen = zlen // 2
    value_points = -1 * 0* np.ones((zlen,1))
    value_histor = -1 * 0 * np.ones((zlen, 1))
    slice_width = 1
    mid_points = np.zeros((zlen,2))

    delta = 10

    coof_zeros = 0

    score_map = np.zeros(edge_map.shape)
    for i in range(0, zlen):
        pstart, pend = pstart_rot, pend_rot

        # print(pstart[0] + i * normal[0])
        # print(pstart[1] + i * normal[1])
        pstart = (pstart[0] + i * normal[0], pstart[1] + i * normal[1])
        pend = (pend[0] + i * normal[0], pend[1] + i * normal[1])
        # print("the start" ,pstart,"the end",pend)
        p1_mid = (int((pend[0] + pstart[0]) / 2), int((pend[1] + pstart[1]) / 2))
        points = [pstart, pend]
        # print("The points are " ,points)
        x_coords, y_coords = zip(*points)
        # print(x_coords,y_coords)


        if points[0][0] == points[1][0]:  # which mean the slope undefined !

            # print("There is no Slope",points)
            y_coordinates = np.linspace(np.min(y_coords).astype(int), np.max(y_coords).astype(int), num=50)
            x_coordinates = points[0][0] * np.ones((1,len(y_coordinates)))[0]


        else:
            a = np.array(points)
            ind = np.lexsort((a[:, 0], a[:, 1]))
            # print(a[ind])
            x_coords = a[ind][:, 0].astype(int)
            y_coords = a[ind][:, 1].astype(int)
            # print("x_intervals",x_coords)
            # print("y_intervals",y_coords)
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]
            # print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
            x_coordinates = np.linspace(np.min(x_coords).astype(int), np.max(x_coords).astype(int), num=100)
            # x_coordinates = np.arange(np.min(x_coords), np.max(x_coords) + 1,0.05)
            y_coordinates = (m * x_coordinates + c).astype(int)
            # print("x_coordinates", x_coordinates)
            # print("y_coordinates", y_coordinates)

        colStartEnd = x_coordinates.astype(int)
        rowStartEnd = y_coordinates.astype(int)

        if(np.max(rowStartEnd) >= edge_map.shape[0]-2) or  (np.min(rowStartEnd) < 0):
            # print("OutOfBounds Rows",np.max(rowStartEnd),np.min(rowStartEnd))
            # input('Enter your input:')
            score_map = 0
            mid_points = None
            value_points[i,:] = 0
            value_histor[i,:] = 0

        elif(np.max(colStartEnd) >= edge_map.shape[1]-2) or  (np.min(colStartEnd) < 0):
            # print("OutOfBound Cols", np.max(colStartEnd), np.min(colStartEnd))
            # input('Enter your input:')
            score_map = 0
            mid_points = None
            value_points[i,:] = 0
            value_histor[i,:] = 0

        else:
            # print("colStartEnd", colStartEnd)
            # print("rowStartEnd", rowStartEnd)

            score_map[(rowStartEnd[0] + rowStartEnd[-1]) // 2, (colStartEnd[0] + colStartEnd[-1]) // 2] = np.sum(edge_map[rowStartEnd, colStartEnd])
            line = edge_map[rowStartEnd, colStartEnd]
            cnt_zeros = line[line == 0]

            #Score up for string lines !
            score_line = np.sum(edge_map[rowStartEnd, colStartEnd])
            # We need to punish for Zeroes
            pun_cost = len(cnt_zeros) * coof_zeros
            # Histogram to let the valuesof boundries line before and line after
            line_before = img_gray[rowStartEnd-slice_width, colStartEnd-slice_width]
            line_after = img_gray[rowStartEnd+slice_width, colStartEnd+slice_width]


            hist, bin_edges = np.histogram(line_before-line_after, bins=np.arange(256))
            score_histo = 0 # this will not affect the score !!
            # print("The histogram",score_histo)
            final_score = score_line+pun_cost+score_histo
            value_points[i,:] = final_score
            value_histor[i,:]=hist[0]
            mid_points[i,:] = ((colStartEnd[0] + colStartEnd[-1]) // 2, (rowStartEnd[0] + rowStartEnd[-1]) // 2)



        # if (value_max <= final_score):
        #     # print(p1_mid_max)
        #     value_max = final_score

        #
            # img_plot = edge_map.copy()
            # cv2.line(img_plot, (int(points[0][0]),int(points[0][1])), (int(points[1][0]),int(points[1][1])), 255)
            # cv2.imshow("SnappingLine", img_plot)
            # cv2.waitKey(1)
            # print("score_line",final_score)


        # for x, y in zip(x_coordinates, y_coordinates):
        #     cv2.circle(img_plot, (int(x), int(y)), 2, 255)
        #     cv2.circle(img_plot, p1_mid, 5, 255)
        #     cv2.imshow("SweepLine", img_plot)

            # print(normal)

    return score_map, mid_points, value_points,value_histor

# def findCornerByLinesV3(img,mask,contour):
#
#     src = mask.copy()
#     src = 255 * src.astype(np.uint8)
#     src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
#     # Check if image is loaded successfully
#     if src is None:
#         print("We have Error Image-loading")
#         exit(0)
#
#     # PART 1
#     ##############################################
#     # ROTATE and refine to get nearest axis alignment !!!
#     # Easy Snap and check !!
#     ###################################################
#
#
#     # PART 1
#     ##############################################
#     #  Calc Eg Vectors as initial results !
#     ###################################################
#     # rot_init_ang = -70
#     # print(rot_init_ang)
#     # rotated_src = rotateImage(src, rot_init_ang)
#     # cv2.imshow('source', rotated_src)
#     # cv2.waitKey(0)
#
#     _, cntr, x_len, y_len, list_points, egenvectors, new_dgs = PCA_FindEdgesOrien(src)
#     fusion_edges = contour.copy()
#     canny_edges  = cv2.Canny(fusion_edges.astype(np.uint8), 50, 150)
#
#
#
#     # PART 2
#     ##################################
#     # Sharp  the edges
#     ##################################
#
#     ################################
#     #
#     #  Second Stage : we will sharpen our image in order to acute the edges
#     #                   by using laplacian filter.
#     #
#     #################
#     # do the laplacian filtering as it is
#     # well, we need to convert everything in something more deeper then CV_8U
#     # because the kernel has some negative values,
#     # and we can expect in general to have a Laplacian image with negative values
#     # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
#     kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
#     # so the possible negative number will be truncated
#     imgLaplacian = cv2.filter2D(fusion_edges, cv2.CV_32F, kernel)
#     sharp = np.float32(fusion_edges)
#     # print(imgLaplacian)
#
#     imgResult = sharp - imgLaplacian
#     # convert back to 8bits gray scale
#     imgResult = np.clip(imgResult, 0, 255)
#     imgResult = imgResult.astype('uint8')
#     imgLaplacian = np.clip(imgLaplacian, 0, 255)
#     imgLaplacian = np.uint8(imgLaplacian)
#     # cv2.imshow('Laplace Filtered Image', imgLaplacian)
#     # cv2.imshow('New Sharped Image', imgResult)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # PART 3
#     ##################################
#     # Find Proper Edges and Lines
#     ##################################
#     # For Fusion and the Result of RCF
#
#     if (len(imgResult.shape) > 2):
#         imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
#
#     _, bw = cv2.threshold(imgResult, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#     # Canny Edge for RCF
#     if (len(canny_edges.shape) > 2):
#         canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_BGR2GRAY)
#     _, canny_edges = cv2.threshold(canny_edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#     # cv2.imshow('bw', bw)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     newX, newY = src.shape[1], src.shape[0]
#     bw = cv2.resize(bw, (int(newX), int(newY)))
#     canny_edges = cv2.resize(canny_edges, (int(newX), int(newY)))
#
#
#     edge_map_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
#     Input_images = [bw, canny_edges]
#     final_lines = []
#     egVs = [egenvectors[0], -1 * egenvectors[0]]
#     degs = [new_dgs[0], new_dgs[0]]
#
#     cv2.imwrite("mask.png", src)
#     cv2.imwrite("fusion.png", fusion_edges)
#     cv2.imwrite("bw.png", bw)
#
#     print("Exit 0")
#     exit(0)
#
#     if SHOW_RESULTS is True:
#         plt.figure()
#         plt.axis('off')
#         plt.imshow(src)
#         plt.savefig('Mask.png')
#
#
#         plt.figure()
#         plt.axis('off')
#         plt.imshow(fusion_edges)
#         plt.savefig('fusion.png')
#
#         plt.figure()
#         plt.axis('off')
#         plt.imshow(bw)
#         plt.savefig('bw.png')
#         plt.show()
#
#
#
#     img_plot = img.copy()
#
#     for egenvec, dg in zip(egVs, degs):
#         #######################################################
#         # Sweep in one Direction
#         ########################################################
#         index_j = 0
#         T = 10
#         p1_max_arr = np.zeros((len(Input_images), len(range(-T, T)), 2))
#         value_max_arr = np.zeros((len(Input_images), len(range(-T, T)), 1))
#         normal_arr = np.zeros((len(Input_images), len(range(-T, T)), 2))
#         deg1 = dg
#         for inp_img in Input_images:
#
#             #######################################################
#             # Edges + RCF Contour
#             ########################################################
#
#             index_i = 0
#
#             for k in range(-T, T):
#                 #######################################################
#                 # Try Different Angels
#                 ########################################################
#                 # img_plot = rotated_src.copy()
#                 a = egenvec
#                 b = np.empty_like(a)
#                 b[0] = -a[1]
#                 b[1] = a[0]
#
#                 pstart, pend = np.array(list_points[0]), np.array(list_points[1])
#                 ### we can Change for degree per sweep
#                 theta = np.radians(10 / T)
#                 c, s = np.cos(k * theta), np.sin(k * theta)
#                 e = np.array(((c, -s), (s, c)))
#                 normal = np.array([b[0], b[1]])
#                 rotated_normal = np.dot(R, normal)
#                 pstart_rot = np.dot(R, pstart).astype(int)
#                 pend_rot = np.dot(R, pend).astype(int)
#                 print(pstart_rot,pend_rot)
#                 cv2.line(img_plot, (pstart_rot[0],pstart_rot[1]),(pend_rot[0],pend_rot[1]), 255)
#                 cv2.imshow("implot point",img_plot)
#                 cv2.waitKey(0)
#
#                 score_map, p1_max, value_max = Score_line(inp_img, rotated_normal,pstart_rot,pend_rot ,x_len, edge_map_rgb)
#                 img_plot = score_map.copy()
#                 # print(p1_max)
#                 # cv2.circle(img_plot, p1_max,3, (0,255,0))
#                 # cv2.imshow("score_map and point max", img_plot)
#                 # print("The Max Value is ",value_max)
#                 # print("The Mid Point is ", p1_max)
#                 # cv2.waitKey(3)
#
#                 p1_max_arr[index_j, index_i, :] = p1_max
#                 normal_arr[index_j, index_i, :] = rotated_normal
#                 value_max_arr[index_j, index_i, :] = value_max
#                 # print(p1_max_arr)
#                 # print(value_max_arr)
#                 # print(normal_arr)
#                 index_i = index_i + 1
#                 # print(score_map.shape)
#                 #######################
#                 # Plot the result
#                 ######################
#
#                 ### Plot :D
#             plt.figure()
#             plt.plot(np.arange(-T, T), value_max_arr[index_j], 'ro')
#             plt.axis([-T, T, np.min(value_max_arr[index_j]), np.max(value_max_arr[index_j])])
#             plt.show()
#             index_j = index_j + 1
#
#         ######### Let calculate the result of both ############33
#         plt.figure()
#         plt.plot(np.arange(-T, T), value_max_arr[0] + value_max_arr[1], 'ro')
#         plt.axis([-T, T, np.min(value_max_arr[0] + value_max_arr[1]), np.max(value_max_arr[0] + value_max_arr[1])])
#         plt.show()
#
#         score_total = value_max_arr[0] + value_max_arr[1]
#         max_point = np.argmax(score_total, axis=0)
#         print(max_point)
#
#         print("The first point ", p1_max_arr[0, max_point, :], "The second point", p1_max_arr[1, max_point, :])
#
#         mid_point = (p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2
#         print("The mid_point ", (p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2)
#
#         print(normal_arr[0, max_point, :])
#         max_normal = -normal_arr[0, max_point, :][0][0]
#         b[0] = -normal_arr[0, max_point, :][0][1]
#         b[1] = normal_arr[0, max_point, :][0][0]
#         print(b[0])
#         ps = (int(np.round(mid_point[0, 0] - b[0] * x_len / 2)), int(np.round(mid_point[0, 1] - b[1] * x_len / 2)))
#         pe = (int(np.round(mid_point[0, 0] + b[0] * x_len / 2)), int(np.round(mid_point[0, 1] + b[1] * x_len / 2)))
#
#         final_lines.append([ps, pe])
#         img_plot = edge_map_rgb.copy()
#         cv2.line(img_plot, (ps[0], ps[1]), (pe[0], pe[1]), 255)
#         plt.figure()
#         plt.imshow(img_plot)
#         plt.show()
#
#     egVs = [egenvectors[1], -egenvectors[1]]
#     degs = [new_dgs[1], new_dgs[1]]
#
#     #######################################################
#     # Sweep in one Direction
#     ########################################################
#     for egenvec, dg in zip(egVs, degs):
#         #######################################################
#         # Sweep in one Direction
#         ########################################################
#         index_j = 0
#         T =1
#         p1_max_arr = np.zeros((len(Input_images), len(range(-T, T)), 2))
#         value_max_arr = np.zeros((len(Input_images), len(range(-T, T)), 1))
#         normal_arr = np.zeros((len(Input_images), len(range(-T, T)), 2))
#         deg1 = dg
#         for inp_img in Input_images:
#
#             #######################################################
#             # Edges + RCF Contour
#             ########################################################
#
#             index_i = 0
#
#             for k in range(-T, T):
#                 #######################################################
#                 # Try Different Angels
#                 ########################################################
#                 # img_plot = rotated_src.copy()
#                 a = egenvec
#                 b = np.empty_like(a)
#                 b[0] = -a[1]
#                 b[1] = a[0]
#
#                 pstart, pend = np.array(list_points[2]), np.array(list_points[3])
#                 ### we can Change for degree per sweep
#                 theta = np.radians(10 / T)
#                 c, s = np.cos(k * theta), np.sin(k * theta)
#                 R = np.array(((c, -s), (s, c)))
#                 normal = np.array([b[0], b[1]])
#                 rotated_normal = np.dot(R, normal)
#                 pstart_rot = np.dot(R, pstart).astype(int)
#                 pend_rot = np.dot(R, pend).astype(int)
#                 # print(pstart_rot,pend_rot)
#                 # cv2.line(img_plot, (pstart_rot[0],pstart_rot[1]),(pend_rot[0],pend_rot[1]), 255)
#                 # cv2.imshow("implot",img_plot)
#                 # cv2.waitKey(0)
#
#                 score_map, p1_max, value_max = Score_line(inp_img, rotated_normal,pstart_rot,pend_rot ,y_len)
#                 img_plot = score_map.copy()
#                 # print(p1_max)
#                 # cv2.circle(img_plot, p1_max,3, (0,255,0))
#                 # cv2.imshow("score_map and point max", img_plot)
#                 # print("The Max Value is ",value_max)
#                 # print("The Mid Point is ", p1_max)
#                 # cv2.waitKey(3)
#
#                 p1_max_arr[index_j, index_i, :] = p1_max
#                 normal_arr[index_j, index_i, :] = rotated_normal
#                 value_max_arr[index_j, index_i, :] = value_max
#                 # print(p1_max_arr)
#                 # print(value_max_arr)
#                 # print(normal_arr)
#                 index_i = index_i + 1
#                 # print(score_map.shape)
#                 #######################
#                 # Plot the result
#                 ######################
#
#                 ### Plot :D
#             plt.figure()
#             plt.plot(np.arange(-T, T), value_max_arr[index_j], 'ro')
#             plt.axis([-T, T, np.min(value_max_arr[index_j]), np.max(value_max_arr[index_j])])
#             plt.show()
#             index_j = index_j + 1
#
#         ######### Let calculate the result of both ############33
#         plt.figure()
#         plt.plot(np.arange(-T, T), value_max_arr[0] + value_max_arr[1], 'ro')
#         plt.axis([-T, T, np.min(value_max_arr[0] + value_max_arr[1]), np.max(value_max_arr[0] + value_max_arr[1])])
#         plt.show()
#
#         # score_total = value_max_arr[0] + value_max_arr[1]
#         score_total = value_max_arr[0]
#         max_point = np.argmax(score_total, axis=0)
#         print(max_point)
#
#         print("The first point ", p1_max_arr[0, max_point, :], "The second point", p1_max_arr[1, max_point, :])
#
#         mid_point = (p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2
#         print("The mid_point ", (p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2)
#
#         print(normal_arr[0, max_point, :])
#         max_normal = -normal_arr[0, max_point, :][0][0]
#         b[0] = -normal_arr[0, max_point, :][0][1]
#         b[1] = normal_arr[0, max_point, :][0][0]
#         print(b[0])
#         ps = (int(np.round(mid_point[0, 0] - b[0] * x_len / 2)), int(np.round(mid_point[0, 1] - b[1] * x_len / 2)))
#         pe = (int(np.round(mid_point[0, 0] + b[0] * x_len / 2)), int(np.round(mid_point[0, 1] + b[1] * x_len / 2)))
#
#         final_lines.append([ps, pe])
#         img_plot1 = canny_edges.copy()
#         img_plot2 = bw.copy()
#         cv2.line(img_plot1, (ps[0], ps[1]), (pe[0], pe[1]), 255)
#         cv2.line(img_plot2, (ps[0], ps[1]), (pe[0], pe[1]), 255)
#         plt.figure()
#         plt.imshow(img_plot1)
#         plt.show()
#
#         plt.figure()
#         plt.imshow(img_plot2)
#         plt.show()
#
#     print(final_lines)
#
#     p1_in = final_lines[0][0]
#     p2_in = final_lines[0][1]
#     p3_in = final_lines[2][0]
#     p4_in = final_lines[2][1]
#
#     p_inter1 = intersection2Lines(p1_in, p2_in, p3_in, p4_in)
#     print(p_inter1)
#
#     p1_in = final_lines[0][0]
#     p2_in = final_lines[0][1]
#     p3_in = final_lines[3][0]
#     p4_in = final_lines[3][1]
#
#     p_inter2 = intersection2Lines(p1_in, p2_in, p3_in, p4_in)
#     print(p_inter2)
#
#     p1_in = final_lines[1][0]
#     p2_in = final_lines[1][1]
#     p3_in = final_lines[2][0]
#     p4_in = final_lines[2][1]
#
#     p_inter3 = intersection2Lines(p1_in, p2_in, p3_in, p4_in)
#     print(p_inter3)
#
#     p1_in = final_lines[1][0]
#     p2_in = final_lines[1][1]
#     p3_in = final_lines[3][0]
#     p4_in = final_lines[3][1]
#
#     p_inter4 = intersection2Lines(p1_in, p2_in, p3_in, p4_in)
#     print(p_inter4)
#
#     img_plot2 = img.copy()
#     for i in [p_inter1, p_inter2, p_inter3, p_inter4]:
#         cv2.circle(img, (int(i[0]), int(i[1])), 3, (255, 255, 0), 2)
#         cv2.imshow("The points", img_plot2)
#         cv2.waitKey(0)
#
#     corners = (np.array(np.array([p_inter1, p_inter2, p_inter3, p_inter4])))
#     corners = corners.reshape(4, 2, 1)
#     print(corners)

def  rotate_around_point(point, radians, origin=(0, 0)):
    """Rotate a point around a given point.

    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy


def proposedLineLSD(img,name_img,mask,img_inputs,folderName=None):

    print(" Start proposedLineLSD")
    print(mask.astype(bool))
    if len(mask.shape)==3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # mask_withoutIner = mask
    lsd_rcf = img_inputs[0]*mask
    cv2.imshow("lsd_rcf",lsd_rcf)
    cv2.waitKey(0)


def findCornerByLinesV5(img,name_img,mask,Input_images,folderName=None):
    print("Start findCornerByLinesV5 ")

    src = mask.copy()
    # print("src dtype",src.dtype,src.shape)
    if (len(src.shape) == 2):
        src = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # src = 255*src.astype(np.uint8)
    # Check if image is loaded successfully
    if src is None:
        print("We have Error Image-loading")
        exit(0)

    # PART 1
    ##############################################
    #  Calc Eg Vectors as initial results !
    ###################################################
    _, cntr, x_len, y_len, list_points, egenvectors, new_dgs = PCA_FindEdgesOrien(src)
    if (len(img.shape) > 2):
        img_gray = img.copy()
        img_plot_lines = img.copy() # This For Drawing in it
    else:
        img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        img_plot_lines = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)


    egVs = [-1 * egenvectors[0],egenvectors[0],-1 * egenvectors[1],egenvectors[1]]
    degs = [new_dgs[0], new_dgs[0],new_dgs[1], new_dgs[1]]
    T = 40
    cnt =0
    isFirstEg = True
    for egenvec, dg in zip(egVs, degs):
        #######################################################
        # Sweep in one Direction
        ########################################################
        index_j = 0
        rangeofT = range(-T, T + 1)

        offset = 2
        if cnt > 1 :
           isFirstEg = False

        if isFirstEg is True:
            p1_max_arr = np.zeros((len(Input_images),  len(range(0, len(rangeofT))), (y_len+offset)//2 , 2))
            value_max_arr = np.zeros((len(Input_images),  len(range(0, len(rangeofT))), (y_len+offset)//2 , 1))
            histo_value = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (y_len + offset) // 2, 1))
            vector_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 2))
            degres_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 1))
        else:


            p1_max_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (x_len + offset) // 2, 2))
            value_max_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (x_len + offset) // 2, 1))
            histo_value = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (x_len + offset) // 2, 1))
            vector_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 2))
            degres_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 1))


        for i in range(0,Input_images.shape[0]):

            #######################################################
            # These Are the Input such RCF,Canny and others
            ########################################################
            inp_img = Input_images[i].copy()

            # cv2.imshow("inp_img",inp_img)
            # cv2.waitKey(0)

            index_i = 0
            for k in range(-T, T + 1):
                #######################################################
                # Try Different Angels
                ########################################################
                a = egenvec
                b = np.empty_like(a)
                b[0] = -a[1]
                b[1] = a[0]
                normal = np.array([b[0], b[1]])

                if isFirstEg is True:
                    pstart, pend = np.array(list_points[0]), np.array(list_points[1])
                else:
                    pstart, pend = np.array(list_points[2]), np.array(list_points[3])

                theta = np.radians(20 / T)

                pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
                pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))
                rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
                rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)
                # print(pstart_rot,pend_rot)
                # cv2.line(img_plot_lines,(int(pstart_rot[0]),int(pstart_rot[1])), (int(pend_rot[0]),int(pend_rot[1])), (193, 244, 34), 2)
                # cv2.imshow("the line",img_plot_lines)
                #
                # rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
                #
                # mid_point = (int(np.round(pstart_rot[0]+ pend_rot[0])/2),int(np.round(pstart_rot[1]+ pend_rot[1])/2))
                # cv2.circle(img_plot_lines, (int(mid_point[0]), int(mid_point[1])), 3,(0, 0, 255), 2)
                # normal_points = int(np.round(mid_point[0] +rotated_normal[0]* 30)), int(
                #     np.round(mid_point[1] + rotated_normal[1] * 30 ))
                #
                # cv2.line(img_plot_lines, (int(mid_point[0]), int(mid_point[1])), (int(normal_points[0]), int(normal_points[1])),
                #          (193, 244, 34), 2)
                if isFirstEg is True:
                    score_map, mid_point, value_points,value_histo = Score_line(inp_img, rotated_normal, pstart_rot, pend_rot,(y_len+offset),img_gray.copy())
                else:
                     score_map, mid_point, value_points, value_histo = Score_line(inp_img, rotated_normal, pstart_rot,
                                                                                  pend_rot, (x_len + offset),
                                                                                  img_gray.copy())

                p1_max_arr[index_j, index_i, :] = mid_point
                histo_value[index_j, index_i, :] = value_histo
                value_max_arr[index_j, index_i, :] = value_points
                degres_arr[index_j, index_i, :] = degrees(k * theta)
                index_i = index_i + 1
            # print("Max ",np.max(value_max_arr[index_j]))
            index_j = index_j + 1




        Value_Max_Re = 0
        for i in value_max_arr:
            ViewRes = np.squeeze(i, axis=2)
            Value_Max_Re = Value_Max_Re + ViewRes
            # plt.figure()
            # plt.imshow(Value_Max_Re)
            # plt.show()

        #####################################################
        #
        #  First Factor : Distance from edge points !
        #
        ####################################################
        if isFirstEg is True:
            pstart, pend = np.array(list_points[2]), np.array(list_points[3])
        else:
            pstart, pend = np.array(list_points[0]), np.array(list_points[1])

        distance_th =50
        for j in p1_max_arr:
            div = j - pstart
            div2 = j - pend

            if (np.sum(LA.norm(div, axis=2)) < np.sum(LA.norm(div2, axis=2))):

                len_cntr = LA.norm(div, axis=2)
                len_cntr =1-((len_cntr - np.min(len_cntr)) / (np.max(len_cntr) - np.min(len_cntr)))
                # plt.figure()
                # plt.imshow(len_cntr)
                # plt.show()
                Value_Max_Re = np.multiply(Value_Max_Re,len_cntr)
                plt.figure()
                plt.imshow(Value_Max_Re)
                plt.show()
            else:

                len_cntr = LA.norm(div2, axis=2)
                len_cntr =1-((len_cntr - np.min(len_cntr)) / (np.max(len_cntr) - np.min(len_cntr)))
                # plt.figure()
                # plt.imshow(len_cntr)
                # plt.show()
                Value_Max_Re = np.multiply(Value_Max_Re, len_cntr)
                plt.figure()
                plt.imshow(Value_Max_Re)
                plt.show()
            break

        #####################################################
        #
        #  Second Factor : Histogram !
        #
        ####################################################

        for hit in histo_value:
            hit = np.squeeze(hit,axis=2)
            # plt.figure()
            # plt.imshow(hit)
            # plt.show()
            hit = 1-((hit - np.min(hit)) / (np.max(hit) - np.min(hit)))
            print(hit)
            plt.figure()
            plt.imshow(hit)
            plt.show()
            Value_Max_Re = np.multiply(Value_Max_Re, hit)
            plt.figure()
            plt.imshow(Value_Max_Re)
            plt.show()
            break;



        print("The max",np.max(Value_Max_Re))
        score = np.squeeze(Value_Max_Re.reshape(1,-1))
        print(score.shape)
        angels_rng = np.squeeze(np.repeat(rangeofT, Value_Max_Re.shape[1]).reshape(1,-1))
        print(angels_rng.shape)

        offsets_rng = np.squeeze(np.tile(range(0,Value_Max_Re.shape[1]), Value_Max_Re.shape[0]).reshape(1, -1))
        print(offsets_rng.shape)

        Best = 5
        score_angle_indcies= list(zip(score.tolist(),angels_rng.tolist(),offsets_rng.tolist()))
        print(score_angle_indcies)
        resultBest=sorted(score_angle_indcies, reverse=True)[:Best]
        # The Best:
        result = np.where(Value_Max_Re ==np.max(Value_Max_Re))
        print('Returned tuple of arrays :', result)
        print('List of Indices of maximum element :', result[0][0])

        # Get 3 Best
        print('Returned tuple of arrays  of "Best" :', resultBest)
        print('List of Indices of maximum element :', resultBest[0])


        # Let's  Try extened the lines
        ext_line = 50
        for value, deg, offset_inc in resultBest:

            if isFirstEg is True:
                pstart, pend = np.array(list_points[0]), np.array(list_points[1])
            else:
                pstart, pend = np.array(list_points[2]), np.array(list_points[3])

            pend1 = pend + egenvec * ext_line
            pstart1 = pstart - egenvec *  ext_line

            if ((pend1[0] - pstart1[0]) ** 2 + (pend1[1] - pstart1[1]) ** 2 < (pend[0] - pstart[0]) ** 2 + (
                    pend[1] - pstart[1]) ** 2):
                pend = pend - egenvec * ext_line
                pstart = pstart + egenvec * ext_line
                # print("Inverse")
            else:

                pend = pend1
                pstart = pstart1



            ### we can Change for degree per sweep
            theta = np.radians(20 / T)
            k = deg
            print("The angle is", k * theta)
            # This work rotated around cntr !
            # print("the radian angle is ", degrees(k * theta))
            pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
            pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))

            a = egenvec
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            normal = np.array([b[0], b[1]])
            rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
            rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)

            for i in range(0, offset_inc):
                pstart, pend = pstart_rot, pend_rot


                pstart = (pstart[0] + i * rotated_normal[0], pstart[1] + i * rotated_normal[1])
                pend = (pend[0] + i * rotated_normal[0], pend[1] + i * rotated_normal[1])

                p1_mid = (int((pend[0] + pstart[0]) / 2), int((pend[1] + pstart[1]) / 2))
                pstart = (int(np.round(pstart[0])), int(np.round(pstart[1])))
                pend = (int(np.round(pend[0])), int(np.round(pend[1])))

            cv2.line(img, pstart, pend, (193, 244, 34), 2)
            cv2.circle(img, p1_mid, 4, (71, 244, 173), -1)

            # cv2.line(img_plot_lines, pstart, pend, (193, 244, 34), 2)
            # cv2.circle(img_plot_lines, p1_mid, 4, (71, 244, 173), -1)

            cv2.imshow("The re", img)
            cv2.waitKey(0)
            # break;

        #
        cnt = cnt + 1
    plt.figure()
    plt.imshow(img)
    plt.show()
    cv2.imwrite(folderName + '/' + str(name_img + now.strftime("%H:%M:%S.png")), img)


def findCornerByLinesV4(img,name_img,mask,img_inputs,folderName=None):
    print("Start findCornerByLinesV4 ")

    src = mask.copy()
    # print("src dtype",src.dtype,src.shape)
    if(len(src.shape)== 2):
         src = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # src = 255*src.astype(np.uint8)
    # Check if image is loaded successfully
    if src is None:
        print("We have Error Image-loading")
        exit(0)

    # PART 1
    ##############################################
    # ROTATE and refine to get nearest axis alignment !!!
    # Easy Snap and check !!
    ###################################################

    # PART 1
    ##############################################
    #  Calc Eg Vectors as initial results !
    ###################################################
    # rot_init_ang = -70
    # print(rot_init_ang)
    # rotated_src = rotateImage(src, rot_init_ang)
    # cv2.imshow('source', rotated_src)
    # cv2.waitKey(0)

    _, cntr, x_len, y_len, list_points, egenvectors, new_dgs = PCA_FindEdgesOrien(src)

    print("x_len ",x_len,"y_len",y_len)
    # newX, newY = src.shape[1], src.shape[0]
    # img_inputs_plus_canny = np.zeros((2 * len(img_inputs), newY, newX))
    # j = 0
    #
    # for img_inp in img_inputs:
    #     img_inp = cv2.resize(img_inp.astype(np.float32), (int(newX), int(newY)))
    #     # img_inp = cv2.resize(img_inp, (int(newX), int(newY)))
    #     img_inputs_plus_canny[j]= img_inp.copy()
    #     canny_edges = cv2.Canny(img_inp.astype(np.uint8), 50, 150)
    #     j=j+1
    #     img_inputs_plus_canny[j] = canny_edges.copy()
    #     j=j+1



    #################
    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255


    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # # so the possible negative number will be truncated
    # for img_inputy in img_inputs_plus_canny:
    #     imgLaplacian = cv2.filter2D(img_inputy, cv2.CV_32F, kernel)
    #     sharp = np.float32(img_inputy)
    # # print(imgLaplacian)
    #
    #     imgResult = sharp - imgLaplacian
    #     # convert back to 8bits gray scale
    #     imgResult = np.clip(imgResult, 0, 255)
    #     imgResult = imgResult.astype('uint8')
    #     imgLaplacian = np.clip(imgLaplacian, 0, 255)
    #     imgLaplacian = np.uint8(imgLaplacian)
    #
    #     print("The shape", imgResult.shape)
    #     if (len(imgResult.shape) > 2):
    #         imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    #     _, bw = cv2.threshold(imgResult, 40, 255, cv2.THRESH_BINARY)
    #     img_inputs_plus_canny.append[bw]
    #     # cv2.imshow('Laplace Filtered Image', imgLaplacian)
    #     # cv2.imshow('New Sharped Image', imgResult)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #
    # # PART 3
    # ##################################
    # # Find Proper Edges and Lines
    # ##################################
    # # For Fusion and the Result of RCF
    # img_inputs_plus_canny_final = []
    # newX, newY = src.shape[1], src.shape[0]
    # for img_inputy in img_inputs_plus_canny:
    #
    #     if (len(img_inputy.shape) > 2):
    #         imgResult = cv2.cvtColor(img_inputy, cv2.COLOR_BGR2GRAY)
    #         _, bw = cv2.threshold(imgResult, 40, 255, cv2.THRESH_BINARY).copy()
    #         bw = cv2.resize(bw, (int(newX), int(newY))).copy()
    #         img_inputs_plus_canny_final.append[bw]
    #     else:
    #         bw = cv2.resize(img_inputy, (int(newX), int(newY))).copy()
    #         img_inputs_plus_canny_final.append[bw]


    # cv2.imshow('canny_edges', canny_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # edge_map_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
    Input_images = img_inputs.copy()
    # Input_images = [bw]
    # for i in Input_images:
    #     cv2.imshow('bw', i)
    #     print(i)
    #     cv2.waitKey(0)


    if (len(img.shape) > 2):
        img_gray = img.copy()
        img_plot_lines = img.copy()
    else:
        img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

        img_plot_lines = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)


    # egVs = [-1 * egenvectors[0],egenvectors[0]]
    # degs = [new_dgs[0], new_dgs[0]]
    #
    # for egenvec, dg in zip(egVs, degs):
    #     #######################################################
    #     # Sweep in one Direction
    #     ########################################################
    #
    #     # print("The egenvec", egenvec)
    #     # print("The Degree", dg)
    #     # print("The init point", list_points[0], list_points[1])
    #     index_j = 0
    #     offset = 1
    #     rangeofT = range(-T, T + 1)
    #     print(len(rangeofT))
    #     p1_max_arr = np.zeros((len(Input_images),  len(range(0, len(rangeofT))), (y_len+offset)//2 , 2))
    #     value_max_arr = np.zeros((len(Input_images),  len(range(0, len(rangeofT))), (y_len+offset)//2 , 1))
    #     histo_value = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (y_len + offset) // 2, 1))
    #     vector_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 2))
    #     degres_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 1))
    #     # histo_values=
    #
    #     # if (len(img.shape) > 2):
    #     #     img_gray = img.copy()
    #     #     img_plot_lines = img.copy()
    #     # else:
    #     #     img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #     #
    #     #     img_plot_lines = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #
    #
    #     for i in range(0,Input_images.shape[0]):
    #
    #
    #     # for inp_img in Input_images:
    #
    #         #######################################################
    #         # Edges + RCF Contour
    #         ########################################################
    #         inp_img = Input_images[i].copy()
    #
    #         # cv2.imshow("inp_img",inp_img)
    #         # cv2.waitKey(0)
    #
    #         index_i = 0
    #         for k in range(-T, T + 1):
    #             #######################################################
    #             # Try Different Angels
    #             ########################################################
    #             # img_plot = rotated_src.copy()
    #
    #             # if (len(img.shape) > 2):
    #             #     img_gray = img.copy()
    #             #     img_plot_lines = img.copy()
    #             # else:
    #             #     img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #             #
    #             #     img_plot_lines = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #
    #
    #             a = egenvec
    #             b = np.empty_like(a)
    #             b[0] = -a[1]
    #             b[1] = a[0]
    #             normal = np.array([b[0], b[1]])
    #
    #             pstart, pend = np.array(list_points[0]), np.array(list_points[1])
    #
    #
    #
    #
    #             ### we can Change for degree per sweep
    #
    #             theta = np.radians(20 / T)
    #             # This work rotated around cntr !
    #             # print("the degree angle is ",np.degrees(k * theta))
    #             # print("the radian angle is ", k * theta)
    #             # print("the rdian_ag angle is ", rdian_ag)
    #
    #             pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
    #             pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))
    #
    #             rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
    #             rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)
    #
    #             # print(pstart_rot,pend_rot)
    #             # cv2.line(img_plot_lines,(int(pstart_rot[0]),int(pstart_rot[1])), (int(pend_rot[0]),int(pend_rot[1])), (193, 244, 34), 2)
    #             # cv2.imshow("the line",img_plot_lines)
    #             #
    #             # rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
    #             #
    #             # mid_point = (int(np.round(pstart_rot[0]+ pend_rot[0])/2),int(np.round(pstart_rot[1]+ pend_rot[1])/2))
    #             # cv2.circle(img_plot_lines, (int(mid_point[0]), int(mid_point[1])), 3,(0, 0, 255), 2)
    #             # normal_points = int(np.round(mid_point[0] +rotated_normal[0]* 30)), int(
    #             #     np.round(mid_point[1] + rotated_normal[1] * 30 ))
    #             #
    #             # cv2.line(img_plot_lines, (int(mid_point[0]), int(mid_point[1])), (int(normal_points[0]), int(normal_points[1])),
    #             #          (193, 244, 34), 2)
    #
    #
    #             score_map, mid_point, value_points,value_histo = Score_line(inp_img, rotated_normal, pstart_rot, pend_rot,(y_len+offset),img_gray.copy())
    #             p1_max_arr[index_j, index_i, :] = mid_point
    #             # vector_arr[index_j, index_i, :] = [-1 * rotated_normal[1], rotated_normal[0]]
    #
    #             histo_value[index_j, index_i, :] = value_histo
    #             value_max_arr[index_j, index_i, :] = value_points
    #             degres_arr[index_j, index_i, :] = degrees(k * theta)
    #             index_i = index_i + 1
    #
    #         # cv2.imshow("the line", img_plot_lines)
    #         # cv2.waitKey(0)
    #         print("Max ",np.max(value_max_arr[index_j]))
    #         index_j = index_j + 1
    #
    #     ######## Let calculate the Input Images ############
    #
    #     # if (len(img.shape) > 2):
    #     #     img_gray = img.copy()
    #     #     img_plot_lines = img.copy()
    #     # else:
    #     #     img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #     #
    #     #     img_plot_lines = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #
    #     Value_Max_Re = 0
    #     for i in value_max_arr:
    #         ViewRes = np.squeeze(i, axis=2)
    #         Value_Max_Re = Value_Max_Re + ViewRes
    #         plt.figure()
    #         plt.imshow(Value_Max_Re)
    #         plt.show()
    #
    #     print("The max",np.max(Value_Max_Re))
    #
    #     score = np.squeeze(Value_Max_Re.reshape(1,-1))
    #     print(score.shape)
    #     angels_rng = np.squeeze(np.repeat(rangeofT, Value_Max_Re.shape[1]).reshape(1,-1))
    #     print(angels_rng.shape)
    #
    #     offsets_rng = np.squeeze(np.tile(range(0,Value_Max_Re.shape[1]), Value_Max_Re.shape[0]).reshape(1, -1))
    #     print(offsets_rng.shape)
    #
    #     Best = 5
    #     score_angle_indcies= list(zip(score.tolist(),angels_rng.tolist(),offsets_rng.tolist()))
    #     print(score_angle_indcies)
    #     resultBest=sorted(score_angle_indcies, reverse=True)[:Best]
    #     # The Best:
    #     result = np.where(Value_Max_Re ==np.max(Value_Max_Re))
    #     print('Returned tuple of arrays :', result)
    #     print('List of Indices of maximum element :', result[0][0])
    #
    #     # Get 3 Best
    #     print('Returned tuple of arrays  of "Best" :', resultBest)
    #     print('List of Indices of maximum element :', resultBest[0])
    #
    #
    #     ######### Without HISTOGRAM ###########
    #
    #     for value, deg, offset_inc in resultBest:
    #
    #         pstart, pend = np.array(list_points[0]), np.array(list_points[1])
    #         ### we can Change for degree per sweep
    #         theta = np.radians(20 / T)
    #         k = deg
    #         print("The angle is", k * theta)
    #         # This work rotated around cntr !
    #         # print("the radian angle is ", degrees(k * theta))
    #         pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
    #         pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))
    #
    #         a = egenvec
    #         b = np.empty_like(a)
    #         b[0] = -a[1]
    #         b[1] = a[0]
    #         normal = np.array([b[0], b[1]])
    #         rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
    #         rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)
    #
    #         color_line = tuple((255 * np.random.rand(1, 3)).astype(int)[0])
    #         color_circle = tuple((255 * np.random.rand(1, 3)).astype(int)[0])
    #
    #         for i in range(0, offset_inc):
    #             pstart, pend = pstart_rot, pend_rot
    #             pstart = (pstart[0] + i * rotated_normal[0], pstart[1] + i * rotated_normal[1])
    #             pend = (pend[0] + i * rotated_normal[0], pend[1] + i * rotated_normal[1])
    #
    #             p1_mid = (int((pend[0] + pstart[0]) / 2), int((pend[1] + pstart[1]) / 2))
    #             pstart = (int(np.round(pstart[0])), int(np.round(pstart[1])))
    #             pend = (int(np.round(pend[0])), int(np.round(pend[1])))
    #
    #         cv2.line(img_plot_lines, pstart, pend, (193, 244, 34), 2)
    #         cv2.circle(img_plot_lines, p1_mid, 4, (71, 244, 173), -1)
    #         cv2.imshow("The re", img_plot_lines)
    #         cv2.waitKey(0)
    #         break;

    #     # if (len(img.shape) > 2):
    #     #     img_plot_lines = img.copy()
    #     # else:
    #     #     img_plot_lines = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #     #
    #     # histo_Map = 1
    #     #
    #     # for i in histo_value:
    #     #     ViewRes = np.squeeze(i, axis=2)
    #     #     # ViewRes = 255 * (1-((ViewRes - np.min(ViewRes)) / (np.max(ViewRes) - np.min(ViewRes))))
    #     #     ViewRes = 255*ViewRes
    #     #     Value_Max_Histo = np.multiply(histo_Map, ViewRes)
    #     #     plt.figure()
    #     #     plt.imshow(ViewRes)
    #     #     plt.show()
    #     #
    #     #     break; ## all of matrix are same we need to take one !
    #     #
    #     # print()
    #     # print("Histogram")
    #     # print("The max of histogram", np.max (Value_Max_Histo))
    #     # print("Max indcies",np.where(Value_Max_Histo ==  np.max(Value_Max_Histo)))
    #     # print()
    #     # print("The histogram for candidate points")
    #     # resultBestWithHisto = np.zeros(ViewRes.shape)
    #     # resultBestHist_list = []
    #     # for value, deg, offset_inc  in resultBest:
    #     #     resultBestWithHisto[deg+T,offset_inc] = Value_Max_Histo[deg+T,offset_inc] * value
    #     #     resultBestHist_list.append((Value_Max_Histo[deg+T,offset_inc] * value,deg,offset_inc))
    #     # plt.figure()
    #     # plt.imshow(resultBestWithHisto)
    #     # plt.show()
    #     # resultBestHist_list = sorted(resultBestHist_list, reverse=True)
    #     # print("Max indcies",np.where(resultBestWithHisto ==  np.max(resultBestWithHisto)))
    #     # print("list of all", resultBestHist_list)
    #     #
    #     # for value,deg,offset_inc in resultBestHist_list:
    #     #
    #     #     pstart, pend = np.array(list_points[0]), np.array(list_points[1])
    #     #     ### we can Change for degree per sweep
    #     #     theta = np.radians(20 /T)
    #     #     k = deg
    #     #     print("The angle is", k * theta)
    #     #     # This work rotated around cntr !
    #     #     # print("the radian angle is ", degrees(k * theta))
    #     #     pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
    #     #     pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))
    #     #
    #     #     a = egenvec
    #     #     b = np.empty_like(a)
    #     #     b[0] = -a[1]
    #     #     b[1] = a[0]
    #     #     normal = np.array([b[0], b[1]])
    #     #     rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
    #     #     rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)
    #     #
    #     #     color_line = tuple((255*np.random.rand(1,3)).astype(int)[0])
    #     #     color_circle = tuple((255*np.random.rand(1,3)).astype(int)[0])
    #     #
    #     #     for i in range(0, offset_inc):
    #     #         pstart, pend = pstart_rot, pend_rot
    #     #         pstart = (pstart[0] + i * rotated_normal[0], pstart[1] + i * rotated_normal[1])
    #     #         pend = (pend[0] + i * rotated_normal[0], pend[1] + i * rotated_normal[1])
    #     #
    #     #         p1_mid = (int((pend[0] + pstart[0]) / 2), int((pend[1] + pstart[1]) / 2))
    #     #         pstart = (int(np.round(pstart[0])),int(np.round(pstart[1])))
    #     #         pend = (int(np.round(pend[0])), int(np.round(pend[1])))
    #     #
    #     #     cv2.line(img_plot_lines, pstart, pend, (193, 244, 34),2)
    #     #     cv2.circle(img_plot_lines,p1_mid,4,(71, 244, 173),-1)
    #     #     cv2.imshow("The re", img_plot_lines)
    #     #     cv2.waitKey(0)
    #     #     break;# we need to take the best one only
    #
    #
    #     # mid_point =
    #
    #
    #
    #     # cv2.circle(img_plot, (mid_point[0], mid_point[1]), 5, (0, 0, 255), 1)
    #     # cv2.imshow("the mid point", img_plot)
    #
    #     # print('The normals ', vector_arr[0, max_point, :], b)
    #
    #
    #
    #     # img_plot = cv2.cvtColor(img_org.copy(),cv2.COLOR_GRAY2BGR)
    #     # cv2.line(img, (ps[0], ps[1]), (pe[0], pe[1]), (255, 0, 0))
    #     # cv2.line(img, (ps_2[0], ps_2[1]), (pe_2[0], pe_2[1]), (0, 0, 255))
    #     # cv2.imshow("The re", img)
    #     # cv2.waitKey(0)
    #     # plt.figure()
    #     # plt.imshow(img)
    #     # plt.show()
    #
    #
    #
    #     # for i in histo_value:
    #     #     ViewRes = np.squeeze(i, axis=2)
    #     #     ViewRes = 255 - (ViewRes - np.min(ViewRes)) / (np.max(ViewRes) - np.min(ViewRes))
    #     #     plt.figure()
    #     #     plt.imshow(ViewRes)
    #     #     plt.show()
    #
    #
    #     # value_max_arr_total = []
    #     #
    #     # for i in range(0, len(Input_images)):
    #     #     if (i == 0):
    #     #         value_max_arr_total = value_max_arr[i]
    #     #     else:
    #     #         value_max_arr_total = value_max_arr_total + value_max_arr[i]
    #     # if SHOW_RESULTS is True:
    #     #     plt.figure()
    #     #     plt.plot(rangeofT, value_max_arr_total, 'ro')
    #     #     plt.axis([np.min(rangeofT), np.max(rangeofT), np.min(value_max_arr_total), np.max(value_max_arr_total)])
    #     #     plt.show()
    #     #     print(value_max_arr_total)
    #     # score_total = value_max_arr_total
    #     # max_point = np.argmax(score_total, axis=0)
    #     # max_point = max_point[0]
    #     # # print("max_point", max_point)
    #     #
    #     # if len(Input_images) > 2:
    #     #     mid_deg = np.array([0, 0])
    #     #     mid_point = np.array([0, 0])
    #     #
    #     # for i in range(0, len(Input_images)):
    #     #     if (i == 0):
    #     #
    #     #         if (len(Input_images) == 1):
    #     #             print("Only One Method ", p1_max_arr[0, max_point, :])
    #     #         mid_point = (p1_max_arr[0, max_point, :]).astype(int)
    #     #         mid_deg = degres_arr[0, max_point, :]
    #     #
    #     #         print("The mid_point ", mid_point)
    #     #
    #     #     elif (len(Input_images) == 2):
    #     #         print("Multiy Method ", p1_max_arr[0, max_point, :], p1_max_arr[1, max_point, :])
    #     #         mid_point = ((p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2).astype(int)
    #     #         print("The mid_point ", (p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2)
    #     #         print("Multiy Method ", degres_arr[0, max_point, :], degres_arr[1, max_point, :])
    #     #         mid_deg = ((degres_arr[0, max_point, :] + degres_arr[1, max_point, :]) // 2).astype(int)
    #     #         print("The mid_point ", (degres_arr[0, max_point, :] + degres_arr[1, max_point, :]) // 2)
    #     #     elif (len(Input_images) > 2):
    #     #
    #     #         print("Multiy Method Up of 2 ", p1_max_arr[0, max_point, :], p1_max_arr[1, max_point, :])
    #     #         mid_point = mid_point + p1_max_arr[i, max_point, :]
    #     #         print("The mid_point Multiy Method Up of 2 ", mid_point // len(Input_images))
    #     #         print("Multiy Method Up of 2")
    #     #         mid_deg = mid_deg + degres_arr[i, max_point, :]
    #     #         print("The mid_point Multiy Method Up of 2  ", mid_deg // len(Input_images))
    #     #
    #     # if len(Input_images) > 2:
    #     #     mid_deg = (mid_deg // len(Input_images)).astype(int)
    #     #     mid_point = (mid_point // len(Input_images)).astype(int)
    #     #
    #     #
    #     # b = rotate_around_point(egenvec, np.radians(mid_deg), (0, 0))
    #     # b = b / np.sqrt(b[0] ** 2 + b[1] ** 2)
    #     #
    #     # if(len(img.shape)>2):
    #     #     img_plot = img.copy()
    #     # else:
    #     #     img_plot = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #     #
    #     # # cv2.circle(img_plot, (mid_point[0], mid_point[1]), 5, (0, 0, 255), 1)
    #     # # cv2.imshow("the mid point", img_plot)
    #     #
    #     # # print('The normals ', vector_arr[0, max_point, :], b)
    #     #
    #     # # print(egenvectors)
    #     # ps = (
    #     # int(np.round(mid_point[0] + egenvec[0] * x_len / 2)), int(np.round(mid_point[1] + egenvec[1] * x_len / 2)))
    #     # pe = (
    #     # int(np.round(mid_point[0] - egenvec[0] * x_len / 2)), int(np.round(mid_point[1] - egenvec[1] * x_len / 2)))
    #     #
    #     # # print(ps, pe)
    #     #
    #     # ps_2 = (int(np.round(mid_point[0] + b[0] * x_len / 2)),
    #     #         int(np.round(mid_point[1] + b[1] * x_len / 2)))
    #     # pe_2 = (int(np.round(mid_point[0] - b[0] * x_len / 2)),
    #     #         int(np.round(mid_point[1] - b[1] * x_len / 2)))
    #     # print(ps, pe)
    #     #
    #     # # img_plot = cv2.cvtColor(img_org.copy(),cv2.COLOR_GRAY2BGR)
    #     # # cv2.line(img, (ps[0], ps[1]), (pe[0], pe[1]), (255, 0, 0))
    #     # cv2.line(img, (ps_2[0], ps_2[1]), (pe_2[0], pe_2[1]), (0, 0, 255))
    #     # # cv2.imshow("The re", img)
    #     # # cv2.waitKey(0)
    #     # # plt.figure()
    #     # # plt.imshow(img)
    #     # # plt.show()
    #


    egVs = [-1*egenvectors[1],egenvectors[1]]
    degs = [new_dgs[1],new_dgs[1]]

    print("The egenvecs", egVs)
    print("The Degrees", degs)
    print("The init point", list_points[2], list_points[3])
    print(egenvectors)
    T = 10
    #######################################################
    # Sweep in second Direction
    ########################################################
    for egenvec, dg in zip(egVs, degs):
        #######################################################
        # Sweep in second Direction
        ########################################################

        print("The egenvec", egenvec)
        print("The Degree", dg)
        print("The init point", list_points[2], list_points[3])

        index_j = 0
        offset = 1
        rangeofT = range(-T, T + 1)
        print(len(rangeofT))
        p1_max_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (x_len + offset) // 2, 2))
        value_max_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (x_len + offset) // 2, 1))
        histo_value = np.zeros((len(Input_images), len(range(0, len(rangeofT))), (x_len + offset) // 2, 1))
        vector_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 2))
        degres_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 1))

        for inp_img in Input_images:

            #######################################################
            # Edges + RCF Contour
            ########################################################

            index_i = 0

            for k in range(-T, T + 1):
                #######################################################
                # Try Different Angels
                ########################################################

                a = egenvec
                b = np.empty_like(a)
                b[0] = -a[1]
                b[1] = a[0]
                # print("the egenv",egenvec,"The normal",b)

                pstart, pend = np.array(list_points[2]), np.array(list_points[3])
                ### we can Change for degree per sweep


                pend1 = pend + egenvec*0
                pstart1 = pstart - egenvec *0

                if((pend1[0]-pstart1[0])**2+(pend1[1]-pstart1[1])**2 < (pend[0]-pstart[0])**2+(pend[1]-pstart[1])**2):
                    pend = pend - egenvec * 0
                    pstart = pstart + egenvec * 0
                    # print("Inverse")
                else:

                    pend = pend1
                    pstart = pstart1



                normal = np.array([b[0], b[1]])

                ### we can Change for degree per sweep
                theta = np.radians(20 / T)

                pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
                pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))
                rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
                rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)


                # print(pstart_rot, pend_rot)
                # cv2.line(img_plot_lines, (int(pstart_rot[0]), int(pstart_rot[1])), (int(pend_rot[0]), int(pend_rot[1])),
                #          (193, 244, 34), 2)
                # cv2.imshow("the line", img_plot_lines)
                #
                #
                # mid_point = (
                # int(np.round(pstart_rot[0] + pend_rot[0]) / 2), int(np.round(pstart_rot[1] + pend_rot[1]) / 2))
                # cv2.circle(img_plot_lines, (int(mid_point[0]), int(mid_point[1])), 3, (0, 0, 255), 2)
                # normal_points = int(np.round(mid_point[0] + rotated_normal[0] * 30)), int(
                #     np.round(mid_point[1] + rotated_normal[1] * 30))
                #
                # cv2.line(img_plot_lines, (int(mid_point[0]), int(mid_point[1])),
                #          (int(normal_points[0]), int(normal_points[1])),
                #          (193, 244, 34), 2)
                #
                #
                #
                # print(pstart_rot, pend_rot)
                # cv2.line(img_plot_lines, (int(pstart_rot[0]), int(pstart_rot[1])), (int(pend_rot[0]), int(pend_rot[1])),
                #          (193, 244, 34), 2)

                # cv2.imshow("img_plot_lines",img_plot_lines)
                # cv2.waitKey(0)
                # continue

                ###
                # print("the radian angle is ", degrees(k * theta))
                # pstart_rot = rotate_around_point(pstart, k * theta, (0, 0))
                # pend_rot = rotate_around_point(pend, k * theta, (0, 0))

                # print("the start",pstart_rot,"the end",pend_rot,"the normal",rotated_normal)

                score_map, mid_point, value_points,value_histo = Score_line(inp_img, rotated_normal, pstart_rot, pend_rot, (x_len+offset),img_gray.copy())
                p1_max_arr[index_j, index_i, :] = mid_point
                # vector_arr[index_j, index_i, :] = [-1 * rotated_normal[1], rotated_normal[0]]
                histo_value[index_j, index_i, :] = value_histo
                value_max_arr[index_j, index_i, :] = value_points
                degres_arr[index_j, index_i, :] = degrees(k * theta)
                index_i = index_i + 1

            # cv2.imshow("the line", img_plot_lines)
            # cv2.waitKey(0)
            print("Max ",np.max(value_max_arr[index_j]))
            index_j = index_j + 1



    ###########################################################################################################

    # Without THE HISTGORAM !

    ###########################################################################################################
        Value_Max_Re = 0
        for i in value_max_arr :
            ViewRes = np.squeeze(i, axis=2)
            Value_Max_Re = Value_Max_Re + ViewRes
            plt.figure()
            plt.imshow(Value_Max_Re)
            plt.show()

        # Value_Max_Re = 0
        # for i in histo_value:
        #     ViewRes = np.squeeze(i, axis=2)
        #     Value_Max_Re = 255*(1-(ViewRes - np.min(ViewRes))/(np.max(ViewRes)-np.min(ViewRes)))
        #     plt.figure()
        #     plt.imshow(ViewRes)
        #     plt.show()
        #     plt.figure()
        #     plt.imshow(Value_Max_Re)
        #     plt.show()
        #     break

        
        print("The max", np.max(Value_Max_Re))
        score = np.squeeze(Value_Max_Re.reshape(1, -1))
        print(score.shape)
        angels_rng = np.squeeze(np.repeat(rangeofT, Value_Max_Re.shape[1]).reshape(1, -1))
        print(angels_rng.shape)

        offsets_rng = np.squeeze(np.tile(range(0, Value_Max_Re.shape[1]), Value_Max_Re.shape[0]).reshape(1, -1))
        print(offsets_rng.shape)

        Best = 5
        score_angle_indcies = list(zip(score.tolist(), angels_rng.tolist(), offsets_rng.tolist()))
        print(score_angle_indcies)
        resultBest = sorted(score_angle_indcies, reverse=True)[:Best]
        # The Best:
        result = np.where(Value_Max_Re == np.max(Value_Max_Re))
        print('Returned tuple of arrays :', result)
        print('List of Indices of maximum element :', result[0][0])

        # Get 3 Best
        print('Returned tuple of arrays  of "Best" :', resultBest)
        print('List of Indices of maximum element :', resultBest[0])


        for value, deg, offset_inc in resultBest:

            pstart, pend = np.array(list_points[2]), np.array(list_points[3])
            ### we can Change for degree per sweep
            theta = np.radians(20 / T)
            k = deg
            print("The angle is", k * theta)
            # This work rotated around cntr !
            # print("the radian angle is ", degrees(k * theta))
            pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
            pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))

            a = egenvec
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            normal = np.array([b[0], b[1]])
            rotated_normal = rotate_around_point(normal,k * theta, (0, 0))
            rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)

            for i in range(0, offset_inc):

                pstart, pend = pstart_rot, pend_rot
                pstart = (pstart[0] + i * rotated_normal[0], pstart[1] + i * rotated_normal[1])
                pend = (pend[0] + i * rotated_normal[0], pend[1] + i * rotated_normal[1])

                p1_mid = (int((pend[0] + pstart[0]) / 2), int((pend[1] + pstart[1]) / 2))
                pstart = (int(np.round(pstart[0])), int(np.round(pstart[1])))
                pend = (int(np.round(pend[0])), int(np.round(pend[1])))

            cv2.line(img_plot_lines, pstart, pend, (193, 244, 34), 2)
            cv2.circle(img_plot_lines, p1_mid, 4, (71, 244, 173), -1)
            cv2.imshow("The re", img_plot_lines)
            cv2.waitKey(0)
            # break;  # we need to take the best one only

        ###########################################################################################################

        # USING THE HISTGORAM !

        ###########################################################################################################
        histo_Map = 0
        for i in histo_value:
            ViewRes = np.squeeze(i, axis=2)
            ViewRes = 255 * (1-((ViewRes - np.min(ViewRes)) / (np.max(ViewRes) - np.min(ViewRes))))
            # ViewRes = 255 * ViewRes
            Value_Max_Histo = histo_Map + ViewRes
            plt.figure()
            plt.imshow(ViewRes)
            plt.show()
        #####################################################################################################
        Max_Results = Value_Max_Re + Value_Max_Histo
        print(Value_Max_Histo)
        plt.figure()
        plt.imshow(ViewRes)
        plt.show()


        # print()
        # print("Histogram")
        # print("The max of histogram", np.max(Value_Max_Histo))
        # print("Max indcies", np.where(Value_Max_Histo == np.max(Value_Max_Histo)))
        # print()
        # print("The histogram for candidate points")
        # print("The max", np.max(Value_Max_Histo))
        #
        # Value_Max_Histo = Value_Max_Re + Value_Max_Histo
        #
        # plt.figure()
        # plt.imshow(Value_Max_Histo)
        # plt.show()
        #
        # score = np.squeeze(Value_Max_Histo.reshape(1, -1))
        # print(score.shape)
        # angels_rng = np.squeeze(np.repeat(rangeofT, Value_Max_Histo.shape[1]).reshape(1, -1))
        # print(angels_rng.shape)
        #
        # offsets_rng = np.squeeze(np.tile(range(0, Value_Max_Histo.shape[1]), Value_Max_Histo.shape[0]).reshape(1, -1))
        # print(offsets_rng.shape)
        #
        # Best = 10
        # score_angle_indcies = list(zip(score.tolist(), angels_rng.tolist(), offsets_rng.tolist()))
        # print(score_angle_indcies)
        # resultBest = sorted(score_angle_indcies, reverse=True)[:Best]
        # print(resultBest)
        #
        #
        #
        # for value, deg, offset_inc in resultBest:
        #     resultBestWithHisto[deg + T, offset_inc] = Value_Max_Histo[deg + T, offset_inc] * value
        #     resultBestHist_list.append((Value_Max_Histo[deg + T, offset_inc] * value, deg, offset_inc))
        # plt.figure()
        # plt.imshow(resultBestWithHisto)
        # plt.show()
        # resultBestHist_list = sorted(resultBestHist_list, reverse=True)
        # print("Max indcies", np.where(resultBestWithHisto == np.max(resultBestWithHisto)))
        # print("list of all", resultBestHist_list)
        #
        #
        #
        # for value, deg, offset_inc in resultBest:
        #
        #     pstart, pend = np.array(list_points[2]), np.array(list_points[3])
        #     ### we can Change for degree per sweep
        #     theta = np.radians(10 / T)
        #     k = deg
        #     print("The angle is", k * theta)
        #     # This work rotated around cntr !
        #     # print("the radian angle is ", degrees(k * theta))
        #     pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
        #     pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))
        #
        #     a = egenvec
        #     b = np.empty_like(a)
        #     b[0] = -a[1]
        #     b[1] = a[0]
        #     normal = np.array([b[0], b[1]])
        #     rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
        #     rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)
        #
        #
        #
        #     for i in range(0, offset_inc):
        #         pstart, pend = pstart_rot, pend_rot
        #         pstart = (pstart[0] + i * rotated_normal[0], pstart[1] + i * rotated_normal[1])
        #         pend = (pend[0] + i * rotated_normal[0], pend[1] + i * rotated_normal[1])
        #
        #         p1_mid = (int((pend[0] + pstart[0]) / 2), int((pend[1] + pstart[1]) / 2))
        #         pstart = (int(np.round(pstart[0])), int(np.round(pstart[1])))
        #         pend = (int(np.round(pend[0])), int(np.round(pend[1])))
        #
        #     cv2.line(img_plot_lines, pstart, pend, (193, 244, 34), 2)
        #     cv2.circle(img_plot_lines, p1_mid, 4, (71, 244, 173), -1)
        #     cv2.imshow("The re", img_plot_lines)
        #     cv2.waitKey(0)
        #     # break;  # we need to take the best one only

    #     ######### Let calculate the result of both ############33
    #
    #     value_max_arr_total = []
    #     for i in range(0, len(Input_images)):
    #         if (i == 0):
    #             value_max_arr_total = value_max_arr[i]
    #         else:
    #             value_max_arr_total = value_max_arr_total + value_max_arr[i]
    #
    #     if SHOW_RESULTS is True:
    #         plt.figure()
    #         plt.plot(rangeofT, value_max_arr_total, 'ro')
    #         plt.axis([np.min(rangeofT), np.max(rangeofT), np.min(value_max_arr_total), np.max(value_max_arr_total)])
    #         plt.show()
    #         print(value_max_arr_total)
    #     score_total = value_max_arr_total
    #     max_point = np.argmax(score_total, axis=0)
    #     max_point = max_point[0]
    #     print("max_point", max_point)
    #
    #     if len(Input_images) > 2:
    #         mid_deg = np.array([0,0])
    #         mid_point=  np.array([0,0])
    #
    #
    #     for i in range(0, len(Input_images)):
    #         if (i == 0):
    #
    #             if (len(Input_images) == 1):
    #                 print("Only One Method ", p1_max_arr[0, max_point, :])
    #             mid_point = (p1_max_arr[0, max_point, :]).astype(int)
    #             mid_deg = degres_arr[0, max_point, :]
    #
    #             print("The mid_point ", mid_point)
    #
    #         elif(len(Input_images)==2):
    #             print("Multiy Method ", p1_max_arr[0, max_point, :], p1_max_arr[1, max_point, :])
    #             mid_point = ((p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2).astype(int)
    #             print("The mid_point ", (p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2)
    #             print("Multiy Method ", degres_arr[0, max_point, :], degres_arr[1, max_point, :])
    #             mid_deg = ((degres_arr[0, max_point, :] + degres_arr[1, max_point, :]) // 2).astype(int)
    #             print("The mid_point ", (degres_arr[0, max_point, :] + degres_arr[1, max_point, :]) // 2)
    #         elif(len(Input_images)>2):
    #
    #             print("Multiy Method Up of 2 ", p1_max_arr[0, max_point, :], p1_max_arr[1, max_point, :])
    #             mid_point = mid_point +  p1_max_arr[i, max_point, :]
    #             print("The mid_point Multiy Method Up of 2 ", mid_point //len(Input_images))
    #             print("Multiy Method Up of 2")
    #             mid_deg = mid_deg + degres_arr[i, max_point, :]
    #             print("The mid_point Multiy Method Up of 2  ", mid_deg//len(Input_images))
    #
    #     if len(Input_images) > 2:
    #         mid_deg =  (mid_deg//len(Input_images)).astype(int)
    #         mid_point= (mid_point//len(Input_images)).astype(int)
    #
    #
    #
    #     # max_normal = -normal_arr[0, max_point, :][0][0]
    #
    #     # print(vector_arr)
    #     # print(degres_arr)
    #     # print(p1_max_arr)
    #
    #     b = rotate_around_point(egenvec, np.radians(mid_deg), (0, 0))
    #     b = b / np.sqrt(b[0] ** 2 + b[1] ** 2)
    #     # b[0] = vector_arr[0, max_point, :][0]
    #     # b[1] = vector_arr[0, max_point, :][1]
    #     if( len(img.shape)>2):
    #         img_plot = img.copy()
    #     else:
    #        img_plot = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    #
    #     # cv2.circle(img_plot, (mid_point[0], mid_point[1]), 5, (0, 0, 255), 1)
    #     # cv2.imshow("the mid point", img_plot)
    #
    #     print('The normals ', vector_arr[0, max_point, :], b)
    #
    #     print(egenvectors)
    #     ps = (
    #         int(np.round(mid_point[0] + egenvec[0] * y_len / 2)), int(np.round(mid_point[1] + egenvec[1] * y_len / 2)))
    #     pe = (
    #         int(np.round(mid_point[0] - egenvec[0] * y_len / 2)), int(np.round(mid_point[1] - egenvec[1] * y_len / 2)))
    #
    #     print(ps, pe)
    #
    #     ps_2 = (int(np.round(mid_point[0] + b[0] * y_len / 2)),
    #             int(np.round(mid_point[1] + b[1] * y_len / 2)))
    #     pe_2 = (int(np.round(mid_point[0] - b[0] * y_len / 2)),
    #             int(np.round(mid_point[1] - b[1] * y_len / 2)))
    #     print(ps, pe)
    #
    #     # img_plot = cv2.cvtColor(img_org.copy(), cv2.COLOR_GRAY2BGR)
    #     # cv2.line(img, (ps[0], ps[1]), (pe[0], pe[1]), (255, 0, 0))
    #     cv2.line(img, (ps_2[0], ps_2[1]), (pe_2[0], pe_2[1]), (0, 0, 255))

    plt.figure()
    plt.imshow(img_plot_lines)
    plt.show()
    cv2.imwrite(folderName+'/'+str(name_img+now.strftime("%H:%M:%S.png")),img)



# Sharp of RCF

def RCF_sharpness(img,newY, newX):
    # PART 1
    print(" Start :  Sharp RCF Results")
    ################### RCF ##################
    rcf_model = RCFBD.RCF_BoundaryOcclusionBoundaryDetector()
    # print("Start  Refinement ;D")
    rcf_inputs = [final, refined_img,lsd_grd] = rcf_model.boundary_detection(img)
    # rcf_inputs = [final, refined_img,] = rcf_model.boundary_detection(img)
    ###### Sharpness Using Laplace ########3

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # so the possible negative number will be truncated
    rcf_sharpnes = np.zeros((len(rcf_inputs), newY, newX)).astype('uint8')
    i = 0
    # print(rcf_inputs)

    for rcfimg in rcf_inputs:
        rcfimg = rcfimg.astype('float32').copy()
        rcfimg = 255 * ((rcfimg - np.min(rcfimg)) / (np.max(rcfimg) - np.min(rcfimg)))
        # print("rcfimg type", rcfimg.shape, rcfimg.dtype)

        imgLaplacian = cv2.filter2D(rcfimg, cv2.CV_32F, kernel)
        sharp = np.float32(rcfimg)


        imgResult = sharp - imgLaplacian
        # convert back to 8bits gray scale
        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype('uint8')
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)
        # print("The shape", imgResult.shape)
        if (len(imgResult.shape) > 2):
            imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)

        imgResult_resized = cv2.resize(imgResult, (int(newX), int(newY)))

        # print(imgResult_resized.dtype)
        # print(rcf_sharpnes[i, :, :].dtype)
        rcf_sharpnes[i, :, :] = imgResult_resized.copy()
        # print(imgResult)
        # cv2.imshow("imgResult", rcf_sharpnes[i,:,:])
        # cv2.waitKey(0)
        i = i + 1

    print("End :  Sharp RCF Results")
    return rcf_sharpnes

def intersectionInImage(image,point):
    #point is in x,y style
    # image is gray maybe we can send size better
    if(image.shape[0] <point[1] or image.shape[1] <point[0]):
        return False

    return True

def FindAllQuads(image,mask):

    print("End :  FindAllQuads")

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):

        # print(c)
        # Calculate the area of each contour
        print("Finding Contour ")
        area = cv2.contourArea(c);
        print(area)


    lsd = cv2.createLineSegmentDetector(0)
    img_rcf_LSD = image.astype(np.uint8)
    # img_CannySobel_LSD = result.astype(np.uint8).copy()
    lines = lsd.detect(img_rcf_LSD)[0]  # Position 0 of the returned tuple are the detected lines

    result = np.zeros(image.shape)
    quads_points = np.zeros((4,3,2))
    for line in lines :
        x0_0 = int(round(line[0][0]))
        y0_0 = int(round(line[0][1]))
        x0_1 = int(round(line[0][2]))
        y0_1 = int(round(line[0][3]))

        p1_in = (x0_0,y0_0)
        p2_in =  ( x0_1,y0_1)
        print(p1_in,p2_in)
        for line1 in lines:
            x1_0 = int(round(line1[0][0]))
            y1_0 = int(round(line1[0][1]))
            x1_1 = int(round(line1[0][2]))
            y1_1 = int(round(line1[0][3]))

            p3_in = (x1_0, y1_0)
            p4_in = (x1_1, y1_1)

            if(p1_in==p3_in and p2_in== p4_in):
                continue;

            print(p3_in,p4_in)
            p_inter01 = intersection2Lines(p1_in, p2_in, p3_in, p4_in)
            print(p_inter01)
            if p_inter01 is not False:
                if intersectionInImage(image,p_inter01) is True:
                    quads_points[0,0,:] = p_inter01

            for line2 in lines:

                x2_0 = int(round(line2[0][0]))
                y2_0 = int(round(line2[0][1]))
                x2_1 = int(round(line2[0][2]))
                y2_1 = int(round(line2[0][3]))

                p5_in = (x2_0, y2_0)
                p6_in = (x2_1, y2_1)

                if (p1_in == p5_in and p2_in == p6_in):
                    continue;

                if (p3_in == p5_in and p4_in == p6_in):
                    continue;


                p_inter02 = intersection2Lines(p1_in, p2_in, p5_in, p6_in)
                if (p_inter02 is not False) and  (intersectionInImage(image,p_inter02)):
                    quads_points[0, 1, :] = p_inter02
                print(p_inter02)

                p_inter12 = intersection2Lines(p3_in, p4_in, p5_in, p6_in)
                if (p_inter12 is not False) and (intersectionInImage(image,p_inter12)):
                    quads_points[1, 1, :] = p_inter12
                print(p_inter12)

                for line3 in lines:

                    x3_0 = int(round(line3[0][0]))
                    y3_0 = int(round(line3[0][1]))
                    x3_1 = int(round(line3[0][2]))
                    y3_1 = int(round(line3[0][3]))

                    p7_in = (x3_0, y3_0)
                    p8_in = (x3_1, y3_1)

                    if (p1_in == p7_in and p2_in == p8_in):
                        continue;

                    if (p3_in == p7_in and p4_in == p8_in):
                        continue;

                    if (p5_in == p7_in and p6_in == p8_in):
                        continue;


                    p_inter03 = intersection2Lines(p1_in, p2_in, p7_in, p8_in)
                    if p_inter02 is not False and (intersectionInImage(image,p_inter03)):
                        quads_points[0, 2, :] = p_inter03

                    p_inter13 = intersection2Lines(p3_in, p4_in, p7_in, p8_in)
                    if p_inter12 is not False :
                        if (intersectionInImage(image,p_inter13)):
                                quads_points[1, 2, :] = p_inter13


                    p_inter23 = intersection2Lines(p5_in, p6_in, p7_in, p8_in)
                    if p_inter12 is not False:
                        quads_points[2, 2, :] = p_inter23

                    quads_points_reshapes = quads_points.reshape(12,2)


                    ploty= image.copy()
                    for i in quads_points_reshapes:


                        cv2.circle(ploty,(int(round(i[0])),int(round(i[1]))),3,255,1)
                    cv2.imshow("res",ploty )
                    cv2.waitKey(0)



def lsd_segment(img_arrays,newY, newX):

     # img_arrays is array of gray images !! or gray image with (1,Xsize,Ysize) !
    print("Start :  Get LSD for RCF Results")

    # Create default parametrization LSD
    lsd = cv2.createLineSegmentDetector(0)
    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    kernel3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

    if(len(img_arrays.shape)==3): # array of grayscale imgs
        img_arrays_plus_LSD = np.zeros((img_arrays.shape[0], newY, newX)).astype('uint8')
        if (img_arrays.shape[0] == 1):
            plot_img = img_arrays[0].copy()

    else: # img
        plot_img = img_arrays.copy()
        img_arrays = np.array([img_arrays])
        img_arrays_plus_LSD = np.zeros((1, newY, newX)).astype('uint8')
        # print(img_arrays.shape,img_arrays.dtype)

    for i in range(0, img_arrays.shape[0]):

            dx = cv2.filter2D(img_arrays[i].astype('float32').copy(), cv2.CV_32F, kernel2)
            dy = cv2.filter2D(img_arrays[i].astype('float32').copy(), cv2.CV_32F, kernel3)
            # dx = ndimage.sobel(rcf_sharpnes[i], 0)  # horizontal derivative
            # dy = ndimage.sobel(rcf_sharpnes[i], 1)  # vertical derivative
            mag = np.hypot(dx, dy)  # magnitude
            mag *= 255.0 / np.max(mag)  # normalize (Q&D)
            # cv2.imshow('sobels', mag)

            # print(mag.shape,mag.dtype)
            ret, thresh1 = cv2.threshold(img_arrays[i], 100, 255, cv2.THRESH_BINARY)
            canny_edges = cv2.Canny(np.uint8(thresh1), 100, 200)
            # cv2.imshow('mag', canny_edges)

            result = np.multiply(canny_edges, mag)
            # cv2.imshow('canny with sobel', result)
            # cv2.waitKey(0)
            img_rcf_LSD = img_arrays[i].copy().astype(np.uint8)
            # img_CannySobel_LSD = result.astype(np.uint8).copy()
            lines_img_rcf_LSD = lsd.detect(img_rcf_LSD)[0]  # Position 0 of the returned tuple are the detected lines
            # print(lines_img_rcf_LSD)
            segment_img_rcf_LSD = np.zeros_like(img_rcf_LSD)
            # lines_img_CannySobel_LSD = lsd.detect( img_CannySobel_LSD)[0]
            # segment_img_CannySobel_LSD = np.zeros_like( img_CannySobel_LSD)


            for dline in lines_img_rcf_LSD:
                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                cv2.line(segment_img_rcf_LSD, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
                if (img_arrays_plus_LSD.shape[0] == 1):
                    cv2.line(plot_img,(x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
          # print(segment_img_rcf_LSD)
            img_arrays_plus_LSD[i] = segment_img_rcf_LSD.copy()
            # for dline in lines_img_CannySobel_LSD:
            #     x0 = int(round(dline[0][0]))
            #     y0 = int(round(dline[0][1]))
            #     x1 = int(round(dline[0][2]))
            #     y1 = int(round(dline[0][3]))
            #     cv2.line(segment_img_CannySobel_LSD, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
            #
            # cv2.imshow('segment_img_rcf_LSD', segment_img_rcf_LSD)
            # # cv2.imshow('segment_img_CannySobel_LSD', segment_img_CannySobel_LSD)
            # cv2.waitKey(0)


    # if(img_arrays_plus_LSD.shape[0]==1):
    #     plt.figure()
    #     plt.imshow(plot_img, cmap='gray', vmin=0, vmax=255)
    #     plt.show()

    print("End :  Get LSD for RCF Results")
    return img_arrays_plus_LSD

def Canny_img(img_arrays,newY, newX):
    print("Start :  Get Canny ")


    if (len(img_arrays.shape) == 3):  # array of grayscale imgs
        img_arrays_plus_Canny = np.zeros((img_arrays.shape[0], newY, newX)).astype('uint8')
        # print(img_arrays_plus_Canny.shape, img_arrays_plus_Canny.dtype)
        if(img_arrays.shape[0]==1):
             plot_img = img_arrays[0].copy()
    else:  # img
        plot_img = img_arrays.copy()
        img_arrays = np.array([img_arrays])
        img_arrays_plus_Canny = np.zeros((1, newY, newX)).astype('uint8')
        # print(img_arrays_plus_Canny.shape, img_arrays_plus_Canny.dtype)

    for i in range(0, img_arrays_plus_Canny.shape[0]):

        ret, thresh1 = cv2.threshold(img_arrays[i], 100, 255, cv2.THRESH_BINARY)
        canny_edges = cv2.Canny(np.uint8(thresh1), 70, 200)
        _, bw = cv2.threshold(canny_edges, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow(' img_arrays_plus_Canny[i] ',  canny_edges )
        # cv2.waitKey(0)
        #
        # cv2.imshow(' bw', bw)
        # cv2.waitKey(0)

        img_arrays_plus_Canny[i] = canny_edges.copy()

    print("End :  Get Canny ")
    return img_arrays_plus_Canny


# Sharpness of Fusion and Refine of RCF + line segment of them !!!!
def demo1(img,mask):

    img_gray = rgb2gray(img)
    # print("Main")
    newX, newY = mask.shape[1], mask.shape[0]

    # PART 1
    print("Part 1 : Start :  Sharp RCF Results")
    ################### RCF ##################
    rcf_model = RCFBD.RCF_BoundaryOcclusionBoundaryDetector()
    # print("Start  Refinement ;D")
    rcf_inputs = [final, refined_img] = rcf_model.boundary_detection(img)

    ###### Sharpness Using Laplace ########3

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # so the possible negative number will be truncated

    rcf_sharpnes = np.zeros((len(rcf_inputs), newY, newX)).astype('uint8')
    i = 0
    print(rcf_inputs)

    for rcfimg in rcf_inputs:

        rcfimg = rcfimg.astype('float32').copy()
        rcfimg = 255 * ((rcfimg - np.min(rcfimg)) / (np.max(rcfimg) - np.min(rcfimg)))
        print("rcfimg type", rcfimg.shape, rcfimg.dtype)

        imgLaplacian = cv2.filter2D(rcfimg, cv2.CV_32F, kernel)
        sharp = np.float32(rcfimg)
        # print(imgLaplacian)
        imgResult = sharp - imgLaplacian
        # convert back to 8bits gray scale
        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype('uint8')
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)
        print("The shape", imgResult.shape)
        if (len(imgResult.shape) > 2):
            imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)

        imgResult_resized = cv2.resize(imgResult, (int(newX), int(newY)))

        print(imgResult_resized.dtype)
        print(rcf_sharpnes[i, :, :].dtype)
        rcf_sharpnes[i, :, :] = imgResult_resized.copy()
        print(imgResult)
        # cv2.imshow("imgResult", rcf_sharpnes[i,:,:])
        # cv2.waitKey(0)
        i = i + 1

    print(rcf_sharpnes)
    print("Part 1 : End :  Sharp RCF Results")

    print()

    # Create default parametrization LSD
    lsd = cv2.createLineSegmentDetector(0)
    # PART 2
    print("Part 2 : Start :  Get LSD for RCF Results")
    ##################################
    # Find Proper Edges and Lines
    ##################################
    # For Fusion and the Result of RCF

    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    kernel3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

    rcf_sharpnes_plus_LSD = np.zeros((2 * len(rcf_inputs), newY, newX)).astype('uint8')

    for i in range(0, rcf_sharpnes.shape[0]):

        rcf_sharpnes_plus_LSD[2 * i] = rcf_sharpnes[i]

        dx = cv2.filter2D(rcf_sharpnes[i].astype('float32').copy(), cv2.CV_32F, kernel2)
        dy = cv2.filter2D(rcf_sharpnes[i].astype('float32').copy(), cv2.CV_32F, kernel3)

        # dx = ndimage.sobel(rcf_sharpnes[i], 0)  # horizontal derivative
        # dy = ndimage.sobel(rcf_sharpnes[i], 1)  # vertical derivative
        mag = np.hypot(dx, dy)  # magnitude
        mag *= 255.0 / np.max(mag)  # normalize (Q&D)
        # cv2.imshow('sobels', mag)

        # print(mag.shape,mag.dtype)
        ret, thresh1 = cv2.threshold(rcf_sharpnes[i], 100, 255, cv2.THRESH_BINARY)
        canny_edges = cv2.Canny(np.uint8(thresh1), 100, 200)
        # cv2.imshow('mag', canny_edges)

        result = np.multiply(canny_edges, mag)
        # cv2.imshow('canny with sobel', result)
        # cv2.waitKey(0)

        img_rcf_LSD = rcf_sharpnes[i].astype(np.uint8).copy()
        img_CannySobel_LSD = result.astype(np.uint8).copy()

        lines_img_rcf_LSD = lsd.detect(img_rcf_LSD)[0]  # Position 0 of the returned tuple are the detected lines
        segment_img_rcf_LSD = np.zeros_like(img_rcf_LSD)
        # lines_img_CannySobel_LSD = lsd.detect( img_CannySobel_LSD)[0]
        # segment_img_CannySobel_LSD = np.zeros_like( img_CannySobel_LSD)

        for dline in lines_img_rcf_LSD:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            cv2.line(segment_img_rcf_LSD, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)

        # print(segment_img_rcf_LSD)
        rcf_sharpnes_plus_LSD[2 * i + 1] = segment_img_rcf_LSD.copy()
        # for dline in lines_img_CannySobel_LSD:
        #     x0 = int(round(dline[0][0]))
        #     y0 = int(round(dline[0][1]))
        #     x1 = int(round(dline[0][2]))
        #     y1 = int(round(dline[0][3]))
        #     cv2.line(segment_img_CannySobel_LSD, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
        #
        # cv2.imshow('segment_img_rcf_LSD', segment_img_rcf_LSD)
        # # cv2.imshow('segment_img_CannySobel_LSD', segment_img_CannySobel_LSD)
        # cv2.waitKey(0)

    print(rcf_sharpnes_plus_LSD)
    print("Part 2 : End :  Get LSD for RCF Results")
    return rcf_sharpnes_plus_LSD

# Sharpness of Fusion and Refine of RCF + line segment of them + line segment of image !!!!
def demo2(img,mask):

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print("Main")
    newX, newY = mask.shape[1], mask.shape[0]
    print(img.shape, img.dtype)
    print()
    rcf_sharpnes = RCF_sharpness(img,newY,newX)
    print(rcf_sharpnes.shape,rcf_sharpnes.dtype)
    print()

    lsd_inputs_rcf = lsd_segment(rcf_sharpnes,newY,newX)
    print(lsd_inputs_rcf.shape, lsd_inputs_rcf.dtype)
    print()

    lsd_inputs_imges = lsd_segment(img_gray, newY, newX)
    print(lsd_inputs_imges.shape, lsd_inputs_imges.dtype)
    print()

    final_results = np.concatenate((rcf_sharpnes, lsd_inputs_rcf,lsd_inputs_imges), axis=0).astype('uint8')
    print(final_results.shape,final_results.dtype)
    print()

    # for i in range(0,final_results.shape[0]):
    #     cv2.imshow("results",final_results[i])
    #     cv2.waitKey(0)

    return final_results

def demo3(img,mask):

    print("Start : Demo 3 ")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    newX, newY = mask.shape[1], mask.shape[0]
    # print("The Shape is ",newX, newY)
    img_gray = cv2.resize(img_gray,(int(newX), int(newY))).copy()

    # DATA FROM RAW IMAGE

    # LSD of Image

    lsd_inputs_imges = lsd_segment(img_gray, newY, newX)

    print(lsd_inputs_imges.shape, lsd_inputs_imges.dtype)
    print()

    final_results = lsd_inputs_imges.astype('uint8')


    #
    # # Canny of Image
    # raw_img_canny = Canny_img(img_gray, newY, newX)
    # print(raw_img_canny.shape, raw_img_canny.dtype)
    # print()
    #
    # LSD of Canny Image
    # lsd_img_canny = lsd_segment(raw_img_canny, newY, newX)
    # print(lsd_img_canny.shape, lsd_img_canny.dtype)
    # print()


    # DATA FROM Contour
    rcf_sharpnes = RCF_sharpness(img, newY, newX)
    print("The Shape of ",rcf_sharpnes.shape, rcf_sharpnes.dtype)
    print()

    # img_canny_rcf_sharpnes = Canny_img(rcf_sharpnes[0:2], newY, newX)
    img_canny_rcf_sharpnes = Canny_img(rcf_sharpnes, newY, newX)
    print(img_canny_rcf_sharpnes.shape, img_canny_rcf_sharpnes.dtype)
    print()

    # we can apply close
    # lsd_inputs_rcf_sharpnes = lsd_segment(rcf_sharpnes[0],newY,newX)
    lsd_inputs_rcf_sharpnes = lsd_segment(rcf_sharpnes, newY, newX)
    print(lsd_inputs_rcf_sharpnes.shape, lsd_inputs_rcf_sharpnes.dtype)
    print()
    #
    lsd_inputs_img_canny_rcf_sharpnes = lsd_segment(img_canny_rcf_sharpnes, newY, newX)
    print(lsd_inputs_img_canny_rcf_sharpnes.shape, lsd_inputs_img_canny_rcf_sharpnes.dtype)
    print()

    final_results = np.concatenate((final_results,img_canny_rcf_sharpnes,lsd_inputs_rcf_sharpnes,lsd_inputs_img_canny_rcf_sharpnes),axis=0).astype('uint8')

    print("final_results shape ",final_results.shape)

    add_mat =  np.zeros(final_results[0].shape)
    mul_mat = np.ones(final_results[0].shape)
    for i in range(0, final_results.shape[0]):
        result = final_results[i].copy()
        print(result.shape)
        result = 255 * (result - np.min(result)) / (np.max(result) - np.min(result))
        result[result > 0] = 255
        plt.imshow(result)
        plt.show()
        # img = result
        # plt.hist(img.ravel(), 256, [0, 256]);
        # plt.show()
        add_mat = add_mat + result
        mul_mat = np.multiply(mul_mat,result)


    print(add_mat)
    norm_add_mat = 255*(add_mat*1/(255*final_results.shape[0]))
    print(norm_add_mat.astype("uint8"))
    plt.imshow(norm_add_mat.astype("uint8"))
    plt.show()

    print(mul_mat)
    plt.imshow(mul_mat.astype("uint8"))
    plt.show()

    final_results = np.expand_dims(norm_add_mat, axis=0)
    # exit(0)
    return final_results

    ###############################################
    kernel = np.ones((3, 3), np.uint8)
    lsd_inputs_img_canny_rcf_sharpnes[lsd_inputs_img_canny_rcf_sharpnes>0]=255
    lsd_inputs_img_canny_rcf_sharpnes = cv2.erode(lsd_inputs_img_canny_rcf_sharpnes, kernel, iterations=1)

    # final_results = rcf_sharpnes.copy()
    final_results = np.concatenate((lsd_inputs_imges,np.array([rcf_sharpnes[0]])), axis=0).astype('uint8')
    print("The shape of",final_results.shape)
    result  =  np.multiply(rcf_sharpnes[0],lsd_inputs_imges[0])

    result = (result-np.min(result))/(np.max(result)-np.min(result))
    # cv2.imshow("result",result)
    # cv2.waitKey(0)
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel, iterations=1)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    result = 255*(result - np.min(result)) / (np.max(result) - np.min(result))
    result[result>0]= 255
    # cv2.imshow("result", result)
    # cv2.waitKey(0)

    # final_results = np.expand_dims(result, axis=0).astype('uint8')
    # print(final_results.shape)
    final_results = np.concatenate((np.expand_dims(result, axis=0),np.expand_dims(img_canny_rcf_sharpnes[0], axis=0),
                                    lsd_inputs_img_canny_rcf_sharpnes,np.expand_dims(lsd_inputs_imges, axis=0)), axis=0).astype('uint8')
    print("The shape of", final_results.shape)

    for i in range(0,final_results.shape[0]):
        print("s")
        cv2.imshow("results",final_results[i])
        img = final_results[i]
        plt.hist(img.ravel(), 256, [0, 256]);
        plt.show()
        # cv2.waitKey(0)


    #
    # result  =  np.multiply(1,lsd_inputs_imges[0])
    # cv2.imshow("result",result)
    # cv2.waitKey(0)
    #
    #
    # cv2.destroyAllWindows()
    #
    # # print(final_results.shape,final_results.dtype)
    # # print()
    #
    # print("End : Demo 3 ")



    return final_results

def extract_corners2(img,name_img,mask=None,folderName=None):

    flag_snap_line =True
    print("We work with folder",folderName)

    ## Let's Try without threshould !!!

        # if (len(img_inputy.shape) > 2):
        #     imgResult = cv2.cvtColor(img_inputy, cv2.COLOR_BGR2GRAY)
        #     _, bw = cv2.threshold(imgResult, 40, 255, cv2.THRESH_BINARY).copy()
        #     bw = cv2.resize(bw, (int(newX), int(newY))).copy()
        #     img_inputs_plus_canny_final.append[bw]
        # else:
        #     bw = cv2.resize(img_inputy, (int(newX), int(newY))).copy()
        #     img_inputs_plus_canny_final.append[bw]

    # cv2.imshow('canny_edges', canny_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corners = []
    refPt= []
    # Maybe we need step of emphzise edges and remove the contains

    ######################
    #
    # We need emphsize the edges and blur to zero the inners regions !!!
    # refined_img
    ######################

    # #fusion image
    # Edge_boud = np.zeros(img_gray.shape)
    # ret, thresh1 = cv2.threshold(255 * final, 100, 255, cv2.THRESH_BINARY)
    # edges = cv2.Canny(np.uint8(thresh1), 100, 200)
    #
    # newX, newY = mask.shape[1], mask.shape[0]
    # edges = cv2.resize(edges, (int(newX), int(newY)))

    # newX, newY = final.shape[1], final.shape[0]
    # edges = cv2.resize(edges, (int(newX), int(newY)))


    # if SHOW_RESULTS:
    #     plt.figure()
    #     plt.imshow(np.uint8(thresh1), cmap='gray', vmin=0, vmax=255)
    #     plt.figure()
    #     plt.imshow(edges,cmap='gray', vmin=0, vmax=255)
    #     plt.show()
    #
    # #########



    # it's take the points as sample of  maskRCNN mask points
    ## Manual Snapping to Using Mouse !! :D
    if mask is None:
        print("Snapping from Mouse")
        refPt.clear()
        for dir in [Direction.Down, Direction.Left, Direction.Up, Direction.Right]:
            print("Start New Snapping")
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", pick_pionts)
            cv2.imshow("image", edges)
            cv2.waitKey(0)
            print("We have Down Snapping now ", len(refPt))
            snap_nearestEdge(refPt, edges, dir)
            refPt.clear()
        print("End  of our Code")
    ### Snapping Using Detected Mask of MaskRCNN
    else:
        if flag_snap_line is True:

            #  demo1 : INPUT Is consist of refine +fusion and thier line segment !!
            img_inputs = demo3(img,mask)

            if folderName is not None:

                # corners = proposedLineLSD(img,name_img,mask,img_inputs,folderName)
                corners = findCornerByLinesV5(img,name_img,mask,img_inputs,folderName)

            else:
                # corners = proposedLineLSD(img, name_img, mask, img_inputs)
                corners = findCornerByLinesV5(img, name_img, mask, img_inputs)
        else:

            # print("Snapping Points : start code ")
            # print(np.argwhere(255*mask.astype(int)==255))
            un_x = np.unique(np.sort(np.argwhere(255*mask.astype(int)==255)[0::][:,0]))
            un_y = np.unique(np.sort(np.argwhere(255 * mask.astype(int) == 255)[0::][:, 1]))

            print(int(np.mean(un_x)))
            print(int(np.mean(un_y)))
            snp = np.zeros(mask.shape)
            # print(snp.shape)
            snp[int(np.mean(un_x)),un_y] = 255
            snp[un_x,int(np.mean(un_y))] = 255

            points_vert = list(zip((int(np.mean(un_x))*np.ones(un_y.shape)).tolist(),un_y.tolist()))
            points_horz = list(zip(un_x.tolist(),(int(np.mean(un_y))*np.ones(un_x.shape)).tolist()))

            for dir in [Direction.Right,Direction.Up,Direction.Left,Direction.Down]:

                if dir == Direction.Left or dir==Direction.Right:
                    refPt = points_horz.copy()
                    snap_nearestEdge(refPt, edges, dir,Edge_boud,corners)
                    refPt.clear()
                else:
                    refPt = points_vert.copy()
                    snap_nearestEdge(refPt, edges, dir,Edge_boud,corners)
                    refPt.clear()

                if SHOW_RESULTS:

                    plt.imshow(Edge_boud)
                    plt.show()
                    print(corners)

            # print("Snapping Points : End code ")
    ###################################################################
            # print("Snapping Mask in-out : start code ")




    return corners

# // Finds the intersection of two lines, or returns false.
# // The lines are defined by (p1, p2) and (p3, p4).
#//// The lines are defined by (o1, p1) and (o2, p2).
def intersection2Lines(p1,p2,p3,p4):

    x = ((p3[0] - p1[0]),(p3[1] - p1[1]))
    d1 = ((p2[0] - p1[0]),(p2[1] - p1[1]))
    d2 = ((p4[0] - p3[0]),(p4[1] - p3[1]))


    cross = d1[0]*d2[1] - d1[1]*d2[0];
    if (np.abs(cross) < 1e-8):
        return False;

    t1 = (x[0] * d2[1] - x[1] * d2[0])/cross;
    r = (p1[0]+d1[0]*t1,p1[1]+d1[1]*t1)
    return r;


if __name__ == '__main__':

    mask = cv2.imread("Maskmages/Book_Occ0001mask0.png")
    img = cv2.imread("Maskmages/Book_Occ0001img.png")
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    extract_corners2(img,"5957de36856461img",mask,'Outputs/')
    print("Nothing")

