from __future__ import division
import cv2 as cv2
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi,degrees,atan
from numpy.linalg import lstsq
from itertools import product
import random as rng
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
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
def calc_approx_edge(img,deg):
    # WE HAVE ONLY ONE MASK !.!! THIS WILL WORK

    unrot = rotateImage(img,90-deg)

    # cv2.imshow("unrot",unrot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img.shape)

    gray = cv2.cvtColor(unrot, cv2.COLOR_BGR2GRAY)
    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c);
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue

        # print(c.shape)

        rot90 = img.copy()
        corners,x_len,y_len = find_corners(c)
        # for x in corners:
        #     cv2.circle(rot90, x, 3, (255, 255, 255), 2)
        # cv2.imshow("unrot", rot90)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        break

    return unrot,x_len,y_len,90-deg

def rotateImage(image, angle,centr=None):

  if   centr is None :
    centr = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(centr, angle, 1.0)
  print(rot_mat)
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
    # print(eigenvectors)

    # unrotate to get good image :D and calc the initial guess for better and easy calculation
    d1_deg = np.round(degrees(atan2(eigenvectors[0, 0],eigenvectors[0, 1])))
    d2_deg = np.round(degrees(atan2(eigenvectors[1, 0], eigenvectors[1, 1])))

    _, x_len, y_len,_ = calc_approx_edge(img, d1_deg)
    # unrot = rotateImage(img, 90 - deg)
    # print(eigenvectors)
    # print(unrotate_deg,unrotate_deg2)
    #
    # if(unrotate_deg>= 0 and unrotate_deg2>=0 ):
    #   print('Q1 and Q2')
    #   if(unrotate_deg <=unrotate_deg2):
    #         rotTo90_image,x_len, y_len,new_deg  = calc_approx_edge(img,unrotate_deg)
    #   else:
    #       rotTo90_image, x_len, y_len, new_deg = calc_approx_edge(img, unrotate_deg2)
    #
    # elif(unrotate_deg> 0 and unrotate_deg2 < 0 ):
    #     print('Q4 and Q1')
    #     rotTo90_image, x_len, y_len, new_deg = calc_approx_edge(img, unrotate_deg2)
    # else:
    #     rotTo90_image, x_len, y_len, new_deg = calc_approx_edge(img, 90+unrotate_deg2)

    p1 = (int(np.round(cntr[0] + eigenvectors[0, 0] * x_len/2)),  int(np.round(cntr[1] + eigenvectors[0, 1] * x_len/2)))
    p2 = (int(np.round(cntr[0] - eigenvectors[0, 0] * x_len / 2)), int(np.round(cntr[1] - eigenvectors[0, 1] * x_len / 2)))
    p3 = (int(np.round(cntr[0] - eigenvectors[1, 0] * y_len/2)),  int(np.round(cntr[1] - eigenvectors[1, 1] * y_len/2)))
    p4 = (int(np.round(cntr[0] + eigenvectors[1, 0] * y_len / 2)), int(np.round(cntr[1] + eigenvectors[1, 1] * y_len / 2)))



    # print(cntr,p1)
    img_plot = img.copy()
    cv2.line(img_plot, p2, p1, 255)
    cv2.line(img_plot, p4, p3, 255)

    plt.figure()
    plt.imshow(img_plot)
    plt.show()
    # cv2.imshow("The res",img_plot )
    # cv2.waitKey(0)

    return _,cntr, x_len, y_len,[p1,p2,p3,p4],eigenvectors,[d1_deg,d2_deg]
def PCA_FindEdgesOrien(img):

    imageTo90, centr, x_len, y_len = 0,0,0,0
    # Convert image to binary
    ########### This Part is to know orientation and rotate to 90
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        print("Finding Contour ")
        area = cv2.contourArea(c);
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue
            # Draw each contour only for visualisation purposes
            # cv2.drawContours(rotated_src, contours, i, (0, 0, 255), 2);
            # Find the orientation of each shape
        print("getOrientation ")
        _, centr, x_len, y_len,list_point,eg_vectors,new_degs= getOrientation(c, img)
        # print(c.shape)
        break

    return _, centr, x_len, y_len,list_point,eg_vectors,new_degs
import enum
class Direction(enum.Enum):
    Left = 1
    Right = 2
    Down = 3
    Up = 4
def SnapLine(edge_map,direction,len_snp,x_cor,y_cor,img_res=None,egV=None,c=None):
    score_map = np.zeros(edge_map.shape)


    colStartEnd = x_cor.astype(int)
    rowStartEnd = y_cor.astype(int)

    len_snp = len_snp//2
    value_max, value_min = 0, np.inf
    p1, p2 = [], []

    if (direction == Direction.Left):
        edge_map_plt = edge_map.copy()

        # print(max_snap)
        for i in range(0, len_snp):
            if (value_max <= np.sum(edge_map[rowStartEnd, colStartEnd[0]])):
                p1 = (rowStartEnd, colStartEnd[0])
                value_max = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd, colStartEnd[0])
                value_min = np.sum(edge_map[rowStartEnd, colStartEnd[0]])

            score_map[rowStartEnd[len(rowStartEnd) // 2], colStartEnd[0]] = np.sum(
                edge_map[rowStartEnd, colStartEnd[0]])
            # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
            cv2.line(edge_map_plt, (colStartEnd[0], np.min(rowStartEnd)), (colStartEnd[0], np.max(rowStartEnd)), 255)

            # cv2.imshow("edge_map_plt",edge_map_plt)
            # cv2.waitKey(0)
            # print(np.sum(edge_map[rowStartEnd, colStartEnd[0]]))

            colStartEnd = colStartEnd - 1
    if (direction == Direction.Right):
        edge_map_plt = edge_map.copy()

        # print(max_snap)
        for i in range(0, len_snp):

            if (value_max <= np.sum(edge_map[rowStartEnd, colStartEnd[0]])):
                p1 = (rowStartEnd, colStartEnd[0])
                value_max = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd, colStartEnd[0])
                value_min = np.sum(edge_map[rowStartEnd, colStartEnd[0]])

            score_map[rowStartEnd[len(rowStartEnd) // 2], colStartEnd[0]] = np.sum(
                edge_map[rowStartEnd, colStartEnd[0]])
            # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
            # cv2.line(edge_map_plt, (colStartEnd[0], np.min(rowStartEnd)), (colStartEnd[0], np.max(rowStartEnd)), 255)
            # cv2.imshow("DrawLine",edge_map_plt)
            # cv2.waitKey(0)
            # print(np.sum(edge_map[rowStartEnd, colStartEnd[0]]))
            colStartEnd = colStartEnd + 1

        # score_map_norm = 255 * score_map // np.max(score_map)
        # plt.figure()
        # plt.imshow(score_map)
        # plt.show()
        # plt.figure()
        # plt.imshow(np.log(score_map + 1))
        # plt.show()

        # cv2.imshow("score_map score_map_norm", score_map_norm)
        # cv2.waitKey(0)
    if (direction == Direction.Up):
        edge_map_plt = edge_map.copy()

        # print(max_snap)
        for i in range(0, len_snp):

            if (value_max <= np.sum(edge_map[rowStartEnd[0], colStartEnd])):
                p1 = (rowStartEnd[0], colStartEnd)
                value_max = np.sum(edge_map[rowStartEnd[0], colStartEnd])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd[0], colStartEnd)
                value_min = np.sum(edge_map[rowStartEnd[0], colStartEnd])

            score_map[rowStartEnd[0], colStartEnd[len(colStartEnd) // 2]] = np.sum(
                edge_map[rowStartEnd[0], colStartEnd])
            # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
            cv2.line(edge_map_plt, (np.min(colStartEnd), np.min(rowStartEnd[0])),
                     (np.max(colStartEnd), np.max(rowStartEnd[0])), 255)
            # cv2.imshow("DrawLine",edge_map_plt)
            # cv2.waitKey(0)
            # print(np.sum(edge_map[rowStartEnd[0], colStartEnd]))
            rowStartEnd = rowStartEnd - 1

        score_map_norm = 255 * score_map // np.max(score_map)
        # plt.figure()
        # plt.imshow(score_map)
        # plt.show()
        # plt.figure()
        # plt.imshow(np.log(score_map + 1))
        # plt.show()
        # cv2.imshow("score_map score_map_norm", score_map_norm)
        # cv2.waitKey(0)
    if (direction == Direction.Down):
        edge_map_plt = edge_map.copy()

        # print(max_snap)
        for i in range(0, len_snp):


            if (value_max <= np.sum(edge_map[rowStartEnd[0], colStartEnd])):
                p1 = (rowStartEnd[0], colStartEnd)
                value_max = np.sum(edge_map[rowStartEnd[0], colStartEnd])
            elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
                p2 = (rowStartEnd[0], colStartEnd)
                value_min = np.sum(edge_map[rowStartEnd[0], colStartEnd])

            score_map[rowStartEnd[0], colStartEnd[len(colStartEnd) // 2]] = np.sum(
                edge_map[rowStartEnd[0], colStartEnd])
            # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
            cv2.line(edge_map_plt, (np.min(colStartEnd), np.min(rowStartEnd[0])),
                     (np.max(colStartEnd), np.max(rowStartEnd[0])), 255)

            # cv2.imshow("DrawLine", edge_map_plt)
            # cv2.waitKey(0)
            # print(np.sum(edge_map[rowStartEnd[0], colStartEnd]))
            rowStartEnd = rowStartEnd + 1

        # plt.figure()
        # plt.imshow(score_map)
        # plt.show()
        # plt.figure()
        # plt.imshow(np.log(score_map + 1))
        # plt.show()
        # cv2.imshow("score_map score_map_norm", score_map_norm)
        # cv2.waitKey(0)

        # fig = plt.figure()
        # plt.imshow(score_map)
        # fig.savefig('ScoreMap'+str(direction)+'.png', dpi=fig.dpi)
        # plt.show()
        #
        # fig = plt.figure()
        # plt.imshow(np.log(score_map + 1))
        # fig.savefig('ScoreMapLog'+str(direction)+'.png', dpi=fig.dpi)
        # plt.show()

    # Get the indices of maximum element in numpy array


    plt.figure()
    plt.imshow(score_map)
    plt.show()



    if (img_res is not None):

        result = np.where(score_map == np.amax(score_map))

        # print('Returned tuple of arrays :', np.concatenate(result, axis=0))
        # edge_map_plt  = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
        if (direction == Direction.Right or direction == Direction.Left):
            col_max = result[1]
            print(col_max)
            delta = 10
            cv2.line(img_res, (p1[1], p1[0][0] - delta), (p1[1], p1[0][-1] + delta), 255, 3)
        else:
            row_max = result[0]
            delta = 10
            # cv2.line(img_res, (np.min(colStartEnd) - delta, row_max[0]), (np.max(colStartEnd) + delta, row_max[0]),
            #          (255, 0, 0), 3)

            cv2.line(img_res, (p1[1][0]-delta,p1[0] ), ( p1[1][-1]+ delta,p1[0] ),(255, 0, 0), 3)

        fig = plt.figure()
        plt.imshow(img_res)
        fig.savefig('DrawLine' + str(direction) + '.png', dpi=fig.dpi)
        plt.show()

    return p1,p2

    ############## Have to be a lot of optimzation steps for picking the right line !!! ###
    # We take the max ! right for simplicity

    # score_map = np.zeros(edge_map.shape)
    # xStartEnd = y_cor.astype(int)
    # yStartEnd = x_cor.astype(int)
    #
    # print(xStartEnd)
    # print(yStartEnd)
    #
    # value_max,value_min = 0,np.inf
    # p1,p2 = [],[]
    #
    # if(direction==Direction.Up):
    #     edge_map_plt = edge_map.copy()
    #
    #     # print(max_snap)
    #     for i in range(0,len_snp):
    #
    #         # if(value_max <= np.sum(edge_map[rowStartEnd, colStartEnd[0]])):
    #         #     p1 = (rowStartEnd, colStartEnd[0])
    #         #     value_max = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
    #         # elif(np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min ):
    #         #     p2 = (rowStartEnd, colStartEnd[0])
    #         #     value_min = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
    #
    #         # if(value_max <= np.sum(edge_map[rowStartEnd, colStartEnd])):
    #         #     p1 = (rowStartEnd, colStartEnd)
    #         #     value_max = np.sum(edge_map[rowStartEnd, colStartEnd])
    #         # elif(np.sum(edge_map[rowStartEnd, colStartEnd]) < value_min ):
    #         #     p2 = (rowStartEnd, colStartEnd)
    #         #     value_min = np.sum(edge_map[rowStartEnd, colStartEnd])
    #
    #         # score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[len(colStartEnd)//2]]= np.sum(edge_map[rowStartEnd, colStartEnd])
    #         # score_map_line[0,len_snp-1 - i] = np.sum(edge_map[rowStartEnd, colStartEnd])
    #
    #         # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
    #         cv2.line(edge_map_plt,(xStartEnd[0],yStartEnd[0]), (xStartEnd[-1],yStartEnd[-1]), 255)
    #         cv2.imshow("DrawLine",edge_map_plt)
    #         cv2.waitKey(0)
    #         # print(np.sum(edge_map[rowStartEnd, colStartEnd[0]]))
    #
    #         xStartEnd = np.round(egV[0]* + xStartEnd).astype(int)
    #         yStartEnd =  np.round(-egV[1]*2 + yStartEnd).astype(int)
    #
    #         cv2.line(edge_map_plt, (xStartEnd[0], yStartEnd[0]), (xStartEnd[-1], yStartEnd[-1]), 255)
    #         cv2.imshow("DrawLine", edge_map_plt)
    #         cv2.waitKey(0)
    #         # colStartEnd = colStartEnd -1
    #
    # if (direction == Direction.Right):
    #     edge_map_plt = edge_map.copy()
    #
    #     # print(max_snap)
    #     for i in range(0, len_snp):
    #
    #         if (value_max <= np.sum(edge_map[rowStartEnd, colStartEnd[0]])):
    #             p1 = (rowStartEnd, colStartEnd[0])
    #             value_max = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
    #         elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
    #             p2 = (rowStartEnd, colStartEnd[0])
    #             value_min = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
    #
    #         score_map[rowStartEnd[len(rowStartEnd) // 2], colStartEnd[0]] = np.sum(
    #             edge_map[rowStartEnd, colStartEnd[0]])
    #         score_map_line[0,i] = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
    #         # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
    #         cv2.line(edge_map_plt, (colStartEnd[0], np.min(rowStartEnd)), (colStartEnd[0], np.max(rowStartEnd)), 255)
    #         # cv2.imshow("DrawLine",edge_map_plt)
    #         # cv2.waitKey(0)
    #         # print(np.sum(edge_map[rowStartEnd, colStartEnd[0]]))
    #         colStartEnd = colStartEnd + 1
    #
    #     # score_map_norm = 255 * score_map // np.max(score_map)
    #     # plt.figure()
    #     # plt.imshow(score_map)
    #     # plt.show()
    #     # plt.figure()
    #     # plt.imshow(np.log(score_map + 1))
    #     # plt.show()
    #
    #     # cv2.imshow("score_map score_map_norm", score_map_norm)
    #     # cv2.waitKey(0)
    # if (direction == Direction.Up):
    #     edge_map_plt = edge_map.copy()
    #
    #     # print(max_snap)
    #     for i in range(0, len_snp):
    #
    #         if (value_max <= np.sum(edge_map[rowStartEnd[0], colStartEnd])):
    #             p1 = (rowStartEnd[0], colStartEnd)
    #             value_max = np.sum(edge_map[rowStartEnd[0], colStartEnd])
    #         elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
    #             p2 = (rowStartEnd[0], colStartEnd)
    #             value_min =  np.sum(edge_map[rowStartEnd[0], colStartEnd])
    #
    #
    #         score_map[rowStartEnd[0], colStartEnd[len(colStartEnd) // 2]] = np.sum(
    #             edge_map[rowStartEnd[0], colStartEnd])
    #         score_map_line[len_snp - 1 - i,0] = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
    #         # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
    #         cv2.line(edge_map_plt, (np.min(colStartEnd), np.min(rowStartEnd[0])), (np.max(colStartEnd), np.max(rowStartEnd[0])), 255)
    #         # cv2.imshow("DrawLine",edge_map_plt)
    #         # cv2.waitKey(0)
    #         # print(np.sum(edge_map[rowStartEnd[0], colStartEnd]))
    #         rowStartEnd = rowStartEnd -1
    #
    #     score_map_norm = 255 * score_map // np.max(score_map)
    #     # plt.figure()
    #     # plt.imshow(score_map)
    #     # plt.show()
    #     # plt.figure()
    #     # plt.imshow(np.log(score_map + 1))
    #     # plt.show()
    #     # cv2.imshow("score_map score_map_norm", score_map_norm)
    #     # cv2.waitKey(0)
    #
    # if (direction == Direction.Down):
    #     edge_map_plt = edge_map.copy()
    #
    #     # print(max_snap)
    #     for i in range(0, len_snp):
    #
    #         if (value_max <= np.sum(edge_map[rowStartEnd[0], colStartEnd])):
    #             p1 = (rowStartEnd[0], colStartEnd)
    #             value_max = np.sum(edge_map[rowStartEnd[0], colStartEnd])
    #         elif (np.sum(edge_map[rowStartEnd, colStartEnd[0]]) < value_min):
    #             p2 = (rowStartEnd[0], colStartEnd)
    #             value_min = np.sum(edge_map[rowStartEnd[0], colStartEnd])
    #
    #         score_map[rowStartEnd[0], colStartEnd[len(colStartEnd) // 2]] = np.sum(
    #             edge_map[rowStartEnd[0], colStartEnd])
    #         score_map_line[i, 0] = np.sum(edge_map[rowStartEnd, colStartEnd[0]])
    #
    #         # print("The score at point=",score_map[rowStartEnd[len(rowStartEnd)//2] ,colStartEnd[0]])
    #         cv2.line(edge_map_plt, (np.min(colStartEnd), np.min(rowStartEnd[0])),
    #                  (np.max(colStartEnd), np.max(rowStartEnd[0])), 255)
    #         # cv2.imshow("DrawLine", edge_map_plt)
    #         # cv2.waitKey(0)
    #         # print(np.sum(edge_map[rowStartEnd[0], colStartEnd]))
    #         rowStartEnd = rowStartEnd + 1
    #
    #     score_map_norm = 255 * score_map // np.max(score_map)
    #     # plt.figure()
    #     # plt.imshow(score_map)
    #     # plt.show()
    #     # plt.figure()
    #     # plt.imshow(np.log(score_map + 1))
    #     # plt.show()
    #     # cv2.imshow("score_map score_map_norm", score_map_norm)
    #     # cv2.waitKey(0)
    #
    # fig = plt.figure()
    # plt.imshow(score_map)
    # fig.savefig('ScoreMap'+str(direction)+'.png', dpi=fig.dpi)
    # plt.show()
    #
    # fig = plt.figure()
    # plt.imshow(np.log(score_map + 1))
    # fig.savefig('ScoreMapLog'+str(direction)+'.png', dpi=fig.dpi)
    # plt.show()
    #
    # return 0
    # # Get the indices of maximum element in numpy array
    # if(img_res is not None):
    #
    #     result = np.where(score_map == np.amax(score_map))
    #     # p_max=np.where(score_map_line == np.amax(score_map_line))
    #     # p_min = np.where(score_map_line == np.amin(score_map_line))
    #     # Maybe We have more than one Max we should take one, that are closet to center, index is low
    #     # print("Pmax= ",p_max," and Pmin =",p_min)
    #     # print('Returned tuple of arrays :', np.concatenate(result, axis=0))
    #     # edge_map_plt  = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
    #
    #     if(direction == Direction.Right or direction==Direction.Up):
    #         col_max = result[1]
    #         delta = 10
    #         cv2.line(img_res, (col_max, np.min(rowStartEnd) - delta), (col_max, np.max(rowStartEnd) + delta),
    #                  (255, 0, 0), 3)
    #     else:
    #         row_max = result[0]
    #         delta = 10
    #         cv2.line(img_res, (np.min(colStartEnd) - delta, row_max), (np.max(colStartEnd) + delta, row_max),
    #                  (255, 0, 0), 3)
    #
    #     fig = plt.figure()
    #     plt.imshow(img_res)
    #     # fig.savefig('DrawLine'+str(direction)+'.png', dpi=fig.dpi)
    #     plt.show()
    #
    #
    # return p1,p2
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
def is_point_on_line2(p1, p2, q):
    if (p1[0] == p2[0]) and (p1[1] == p2[1]):
        p1[0] -= 0.00001

    U = ((q[0] - p1[0]) * (p2[0] - p1[0])) + ((q[1] - p1[1]) * (p2[1] - p1[1]))
    Udenom = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
    U /= Udenom

    r = [0, 0]
    r[0] = p1[0] + (U * (p2[0] - p1[0]))
    r[1] = p1[1] + (U * (p2[1] - p1[1]))

    return r
def calculate_line_score(line, contours, cols, M=5, img=None, SHOW_RESULTS=False):
    sum = 0

    if SHOW_RESULTS:
        draw_img = img.copy()

    line = [line[0], line[1], line[2], line[3]]

    # print(line, file=sys.stderr)

    if np.abs(line[0]) < 1e-05:
        line[0] = 0.0001

    if np.abs(line[1]) < 1e-05:
        line[1] = 0.0001

    lefty = int((-line[2] * line[1] / line[0]) + line[3])
    righty = int(((img.shape[1] - line[2]) * line[1] / line[0]) + line[3])


    # print(line, file=sys.stderr)

    if SHOW_RESULTS:
        cv2.line(draw_img, (img.shape[1] - 1, righty), (0, lefty), (100, 0, 255), 1)

    for i in range(len(contours)):
        # not efficient #todo Change
        X = []
        y = []
        for k in range(i - M, i + M):
            k = k % len(contours)
            X.append(contours[k][0])
            y.append(contours[k][1])
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        reg = linear_model.LinearRegression().fit(X, y)

        n = np.asarray([reg.coef_[0][0], 1])
        n = n / np.linalg.norm(n)

        n = np.asarray([-1 * n[0], n[1]])

        projected = is_point_on_line2((cols - 1, righty), (0, lefty), contours[i])


        dist = np.linalg.norm(contours[i] - projected)

        ori = np.dot(np.asarray([line[0], line[1]]).reshape(2), n.reshape(2))

        if dist < 50 and np.abs(ori) < 0.5:
            # print('ori: {}'.format(ori),  file=sys.stderr)
            sum += dist
            if SHOW_RESULTS:
                cv2.circle(draw_img, (int(contours[i][0]), int(contours[i][1])), 2, (255, 0, 0), -1)
                cv2.circle(draw_img, (int(projected[0]), int(projected[1])), 2, (255, 255, 0), -1)
                cv2.line(draw_img, (int(contours[i][0]), int(contours[i][1])), (int(projected[0]), int(projected[1])),
                         (255, 0, 255), 1)
    if SHOW_RESULTS:
        cv2.imshow('projected', draw_img)
        cv2.waitKey(1)

    return sum
def extract_4_poly(contours, img, T=10, M=20, alpha=0, beta=0):


    contours = np.asarray(contours.copy(), dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    lines = []
    d_img = img.copy()
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(d_img, [box], 0, (0, 200, 255), 1)

    for i in range(4):
        [vx, vy, x, y] = cv2.fitLine(np.asarray([box[i], box[(i + 1) % 4]]), cv2.DIST_L2, 0, 0.01, 0.01)
        lines.append((vx, vy, x, y))

    best_line = []

    for i in range(4):
        line = lines[i]
        normal = np.asarray([line[0], line[1]])
        theta = np.radians(10 / T)
        min_score = np.inf
        min_line = line

        for k in range(-T, T, 1):
            c, s = np.cos(k * theta), np.sin(k * theta)
            R = np.array(((c, -s), (s, c)))

            rotated_normal = np.dot(R, normal)

            rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)

            for off in range(-M, M, 1):
                n_l = (rotated_normal[0], rotated_normal[1],
                       line[2] + off, line[3] + off)

                lefty = int((-n_l[2] * n_l[1] / n_l[0]) + n_l[3])
                righty = int(((img.shape[1] - n_l[2]) * n_l[1] / n_l[0]) + n_l[3])

                # cv2.line(d_img, (img.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)
                # cv2.imshow('lines', d_img)
                # cv2.waitKey(1)

                score = calculate_line_score(n_l, contours, cols=img.shape[1], img=img, SHOW_RESULTS=True)

                score += alpha * ((k ** 2) * theta) + beta * off

                # print('sum: {}'.format(score), file=sys.stderr)

                if score < min_score:
                    min_score = score
                    min_line = (n_l[0], n_l[1], n_l[2], n_l[3])

        best_line.append(min_line)

    return  best_line


def Score_line(edge_map, normal,pstart_rot,pend_rot,zlen, edge_map_rgb):

        value_max = -1*np.inf
        p1_mid_max=[]
        p1_mid = []
        delta = 10

        coof_zeros = -500

        score_map = np.zeros(edge_map_rgb.shape)
        zlen= zlen//2
        for i in range(0, zlen):
            pstart, pend = pstart_rot, pend_rot
            pstart = (pstart[0] + i * normal[0], pstart[1] + i * normal[1])
            pend = (pend[0] + i * normal[0], pend[1] + i * normal[1])
            # print("the start" ,pstart,"the end",pend)
            p1_mid = (int((pend[0] + pstart[0]) / 2), int((pend[1] + pstart[1]) / 2))
            points = [pstart, pend]
            x_coords, y_coords = zip(*points)
            # print("The lines with  ", points, "and the degree", deg1)
            # img_plot = rotated_src.copy()
            # cv2.line(img_plot, points[0], points[1], 255)
            # cv2.imshow("SnappingLine", img_plot)
            # cv2.waitKey()

            if points[0][0] == points[1][0]:  # which mean the slope undefined !
                y_coordinates = np.arange(y_coords[0], y_coords[1] + 1)
                x_coordinates = points[0][0] * np.ones(len(y_coordinates))
                # print(x_coordinates)
                # print(y_coordinates)
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
                x_coordinates = np.linspace(np.min(x_coords).astype(int), np.max(x_coords).astype(int),num=50)
                # x_coordinates = np.arange(np.min(x_coords), np.max(x_coords) + 1,0.05)
                y_coordinates = (m * x_coordinates + c).astype(int)
                # print("x_coordinates", x_coordinates)
                # print("y_coordinates", y_coordinates)

            colStartEnd = x_coordinates.astype(int)
            rowStartEnd = y_coordinates.astype(int)
            score_map[(rowStartEnd[0]+rowStartEnd[-1]) // 2,(colStartEnd[0]+colStartEnd[-1]) // 2] = np.sum(edge_map[rowStartEnd, colStartEnd])

            line = edge_map[rowStartEnd, colStartEnd]
            cnt_zeros = line[line==0]
            # print(edge_map[rowStartEnd, colStartEnd].shape)
            # print(len(cnt_zeros))
            # print(p1_mid,np.sum(edge_map[rowStartEnd, colStartEnd])+len(cnt_zeros) * coof_zeros)

            if (value_max <= np.sum(edge_map[rowStartEnd, colStartEnd])+len(cnt_zeros) * coof_zeros):
                p1_mid_max = ((colStartEnd[0]+colStartEnd[-1]) // 2, (rowStartEnd[0]+rowStartEnd[-1]) // 2)
                # print(p1_mid_max)
                value_max = np.sum(edge_map[rowStartEnd, colStartEnd]) +len(cnt_zeros) * coof_zeros

            # img_plot = edge_map_rgb.copy()
            # for x, y in zip(x_coordinates, y_coordinates):
            #     cv2.circle(img_plot, (int(x), int(y)), 2, 255)
            #     cv2.circle(img_plot, p1_mid, 5, 255)
            #     cv2.imshow("SweepLine", img_plot)
            #     # print(normal)
            # cv2.waitKey(0)

        return score_map,p1_mid_max,value_max

def rotate_around_point(point, radians, origin=(0, 0)):
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





if __name__ == '__main__':



    img_org = cv2.imread("/home/borak/Results_Snapping/bw.png")

    src = cv2.imread("/home/borak/Results_Snapping/mask.png")

    # PART 1
    ##############################################
    #  Calc Eg Vectors as initial results !
    ###################################################

    rot_init_ang = 0
    print(rot_init_ang)
    rotated_src= rotateImage(src,rot_init_ang)
    cv2.imshow('source', rotated_src)
     # cv2.waitKey(0)

    _, cntr, x_len, y_len,list_points,egenvectors,new_dgs = PCA_FindEdgesOrien(rotated_src)
    # cv2.imshow('rotated_src ', rotated_src )
    print(list_points)
    # imageTo90, cntr, x_len, y_len,list_points,egenvectors,non_dg = PCA_FindEdgesOrien(rotated_src)
    # print("new_dg",new_dg,"non_dg",non_dg)
    # print(list_points)

    print("The center",cntr)
    print(new_dgs)
    p1 = list_points[0]
    p2 = list_points[1]
    p3 = list_points[2]
    p4 = list_points[3]
    print(p1,p2,p3,p4)



    # Lets take the results
    # a = np.array(list_points)
    # ind = np.lexsort((a[:, 0], a[:, 1]))
    #
    # pstart = (a[ind][0][0], a[ind][0][1])
    # pend = (a[ind][-1][0], a[ind][-1][1])
    #
    # x_min,x_max = np.min(a[ind][:,0]),np.max(a[ind][:,0])
    # y_min, y_max = np.min(a[ind][:, 1]), np.max(a[ind][:, 1 ])
    #
    # list_contour = [(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)]
    # x_coords, y_coords = zip(*points)
    # print(a[ind][:,0])

    fusion_edges = cv2.imread("/home/borak/Results_Snapping/bw.png")
    fusion_edges = rotateImage(fusion_edges, rot_init_ang)
    # print(fusion_edges.shape)
    canny_edges = cv2.imread("/home/borak/Results_Snapping/canny_edges.png")
    canny_edges = rotateImage(canny_edges, rot_init_ang)

    img_inputs=[fusion_edges,canny_edges]

    print(img_inputs)


    # PART 2
    ##################################
    #Sharp  the edges
    ##################################


    # print(refine_edges.shape)
    # src[np.all(refine_edges == 255, axis=2)] = 0
    # # Show output image
    # cv2.imshow('Black Background Image', refine_edges)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    ######################
    #
    #  Second Stage : we will sharpen our image in order to acute the edges
    #                   by using laplacian filter.
    #
    #################
    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0
    # to 255
    newX, newY = src.shape[1], src.shape[0]
    img_inputs_plus = np.zeros((2*len(img_inputs),newY,newX))
    j=0
    for img_inp in img_inputs:

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        # so the possible negative number will be truncated
        imgLaplacian = cv2.filter2D(img_inp, cv2.CV_32F, kernel)
        sharp = np.float32(img_inp)
        # print(imgLaplacian)

        imgResult = sharp - imgLaplacian
        # convert back to 8bits gray scale
        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype('uint8')
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)
        # cv2.imshow('Laplace Filtered Image', imgLaplacian)
        # cv2.imshow('New Sharped Image', imgResult)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # PART 3
        ##################################
        # Find Proper Edges and Lines
        ##################################
        # For Fusion and the Result of RCF

        if (len(img_inp.shape) > 2):
            img_inp = cv2.cvtColor(img_inp, cv2.COLOR_BGR2GRAY)
        img_inp = cv2.resize(img_inp, (int(newX), int(newY)))
        _, img_inp = cv2.threshold(img_inp, 40, 255, cv2.THRESH_BINARY)
        img_inputs_plus[j] = img_inp.copy()

        j=j+1

        if(len(imgResult.shape)>2):
           imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
        imgResult = cv2.resize(imgResult, (int(newX), int(newY)))
        _, imgResult = cv2.threshold(imgResult, 40, 255, cv2.THRESH_BINARY)
        img_inputs_plus[j] = imgResult.copy()
        j=j+1

        # # Canny Edge for RCF
        # if (len(canny_edges.shape) > 2):
        #          canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_BGR2GRAY)
        # _, canny_edges = cv2.threshold(canny_edges, 40, 255, cv2.THRESH_BINARY)
    print("The final result")
    print(img_inputs_plus)
    # for img_inp in img_inputs_plus:
    #     cv2.imshow('img_inp', img_inp)
    #     cv2.waitKey(0)
    # print(canny_edges)
    # newX,newY = src.shape[1],src.shape[0]
    # bw = cv2.resize(bw,(int(newX),int(newY)))
    # canny_edges = cv2.resize(canny_edges,(int(newX),int(newY)))
    # img_org = bw.copy()

    # edge_map_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
    Input_images = img_inputs_plus
    # Input_images = [bw]


    lsd = cv2.createLineSegmentDetector(0)
    for imgy in Input_images:
        # Create default parametrization LSD
        print(imgy)
        cv2.imshow("imgy", imgy)
        cv2.waitKey(0)
        # Detect lines in the image
        img_int = imgy.astype(np.uint8).copy()
        print(img_int)
        lines = lsd.detect(img_int)[0]  # Position 0 of the returned tuple are the detected lines

        segment_line = np.zeros_like(img_int)


        for dline in lines:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            cv2.line(segment_line, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)


        # Draw detected lines in the image
        drawn_img = lsd.drawSegments(img_int, lines)

        # Show image
        cv2.imshow("LSD", segment_line)
        cv2.waitKey(0)



    # our result
    img_ploty = img_inputs_plus[0]
    egVs = [egenvectors[0],-1*egenvectors[0]]
    degs =  [new_dgs[0],new_dgs[0]]

    for egenvec,dg in zip(egVs,degs):
         #######################################################
         # Sweep in one Direction
         ########################################################

        print("The egenvec",egenvec)
        print("The Degree",dg)
        print("The init point",list_points[0],list_points[1])
        index_j = 0

        T = 20
        rangeofT = range(-T,T+1)
        print(len(rangeofT))
        p1_max_arr = np.zeros((len(Input_images),len(range(0, len(rangeofT))),2))
        value_max_arr = np.zeros((len(Input_images),len(range(0, len(rangeofT))),1))
        vector_arr = np.zeros((len(Input_images),len(range(0, len(rangeofT))),2))
        degres_arr = np.zeros((len(Input_images),len(range(0, len(rangeofT))),1))
        deg1 = dg
        for inp_img in Input_images:

            #######################################################
            # Edges + RCF Contour
            ########################################################

            index_i =0
            for k in range(-T,T+1):

                #######################################################
                # Try Different Angels
                ########################################################
                # img_plot = rotated_src.copy()
                a = egenvec
                b = np.empty_like(a)
                b[0] = -a[1]
                b[1] = a[0]

                pstart, pend = np.array(list_points[0]),np.array(list_points[1])
                normal = np.array([b[0],b[1]])

                ### we can Change for degree per sweep
                theta = np.radians(10/T)

                # This work rotated around cntr !
                print("the radian angle is ",degrees(k * theta))
                pstart_rot = rotate_around_point(pstart,k * theta,(cntr[0],cntr[1]))
                pend_rot = rotate_around_point(pend,k * theta,(cntr[0],cntr[1]))
                # rotated_normal = rotate_around_point(normal,k * theta,(cntr[0],cntr[1]))
                # rotated_normal = rotated_normal/np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)
                #
                # ###
                # print("the radian angle is ",degrees(k * theta))
                # pstart_rot = rotate_around_point(pstart, k * theta, (0, 0))
                # pend_rot = rotate_around_point(pend, k * theta, (0, 0))
                rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
                rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)

                score_map,p1_max,value_max = Score_line(inp_img, rotated_normal,pstart_rot,pend_rot, y_len,inp_img.copy())
                p1_max_arr[index_j,index_i,:]=p1_max
                vector_arr[index_j,index_i,:] = [-1*rotated_normal[1],rotated_normal[0]]
                value_max_arr[index_j,index_i,:] = value_max
                degres_arr[index_j,index_i,:] = degrees(k * theta)
                index_i = index_i + 1

            ### Plot :D
            plt.figure()
            plt.plot(rangeofT, value_max_arr[index_j], 'ro')
            plt.axis([np.min(rangeofT), np.max(rangeofT), np.min(value_max_arr[index_j]), np.max(value_max_arr[index_j])])
            plt.show()
            index_j = index_j + 1

        ######## Let calculate the both results ############

        value_max_arr_total = []
        plt.figure()

        for i in range(0,len(Input_images)):
            if(i == 0):
                 value_max_arr_total = value_max_arr[i]
            else:
                value_max_arr_total = value_max_arr_total + value_max_arr[i]
        plt.plot(rangeofT, value_max_arr_total, 'ro')
        plt.axis([np.min(rangeofT), np.max(rangeofT), np.min(value_max_arr_total), np.max(value_max_arr_total)])
        plt.show()
        print(value_max_arr_total)
        score_total = value_max_arr_total
        max_point =np.argmax(score_total, axis=0)
        max_point= max_point[0]
        print("max_point",max_point)

        for i in range(0, len(Input_images)):
             if (i == 0):

                 if(len(Input_images)==1):
                    print("Only One Method ",p1_max_arr[0, max_point, :])
                 mid_point = (p1_max_arr[0, max_point, :]).astype(int)
                 mid_deg = degres_arr[0, max_point, :]

                 print("The mid_point ", mid_point)

             else:
                  print("Multiy Method ", p1_max_arr[0, max_point, :], p1_max_arr[1, max_point, :])
                  mid_point = ((p1_max_arr[0, max_point, :]+ p1_max_arr[1, max_point, :]) //2).astype(int)
                  print("The mid_point ",(p1_max_arr[0, max_point, :]+ p1_max_arr[1, max_point, :]) //2)

                  print("Multiy Method ", degres_arr[0, max_point, :], degres_arr[1, max_point, :])
                  mid_deg = ((degres_arr[0, max_point, :]+ degres_arr[1, max_point, :]) // 2).astype(int)
                  print("The mid_point ", (degres_arr[0, max_point, :] + degres_arr[1, max_point, :]) // 2)


        b = rotate_around_point(egenvec, np.radians(mid_deg), (0,0))
        b = b/np.sqrt(b[0]**2+b[1]**2)

        img_plot = img_inputs_plus[0]
        cv2.circle(img_plot, (mid_point[0], mid_point[1]), 5, (0,0,255),1)
        cv2.imshow("the mid point",img_plot)

        print('The normals ' , vector_arr[0, max_point, :],b)

        print(egenvectors)
        ps = (int(np.round(mid_point[0] + egenvec[0] * x_len/2)),  int(np.round(mid_point[1] + egenvec[1] * x_len/2)))
        pe = (int(np.round(mid_point[0] - egenvec[0] * x_len / 2)), int(np.round(mid_point[1] - egenvec[1] * x_len / 2)))


        print(ps,pe)

        ps_2 = (int(np.round(mid_point[0] + b[0] * x_len / 2)),
               int(np.round(mid_point[1] + b[1] * x_len / 2)))
        pe_2 = (int(np.round(mid_point[0] - b[0] * x_len / 2)),
               int(np.round(mid_point[1] - b[1] * x_len / 2)))
        print(ps, pe)

        # img_plot = cv2.cvtColor(img_org.copy(),cv2.COLOR_GRAY2BGR)
        cv2.line(img_ploty, (ps[0],ps[1]),(pe[0],pe[1]), (255,0,0))
        cv2.line(img_ploty, (ps_2[0], ps_2[1]), (pe_2[0], pe_2[1]), (0, 0, 255))
        cv2.imshow("The re",img_ploty)
        cv2.waitKey(0)



    egVs = [egenvectors[1], -1*egenvectors[1]]
    degs = [ new_dgs[1], new_dgs[1]]
    #######################################################
    # Sweep in second Direction
    ########################################################
    for egenvec, dg in zip(egVs, degs):
        #######################################################
        # Sweep in second Direction
        ########################################################

        # print("The egenvec", egenvec)
        # print("The Degree", dg)
        # print("The init point", list_points[2], list_points[3])
        index_j = 0

        T = 20
        rangeofT = range(-T, T+1)
        p1_max_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 2))
        value_max_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 1))
        vector_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 2))
        degres_arr = np.zeros((len(Input_images), len(range(0, len(rangeofT))), 1))
        deg1 = dg
        for inp_img in Input_images:

            #######################################################
            # Edges + RCF Contour
            ########################################################

            index_i = 0

            for k in range(-T, T+1):
                #######################################################
                # Try Different Angels
                ########################################################
                img_plot = rotated_src.copy()
                a = egenvec
                b = np.empty_like(a)
                b[0] = -a[1]
                b[1] = a[0]

                pstart, pend = np.array(list_points[2]), np.array(list_points[3])
                ### we can Change for degree per sweep

                normal = np.array([b[0], b[1]])

                ### we can Change for degree per sweep
                theta = np.radians(10 / T)
                #
                # c, s = np.cos(k * theta), np.sin(k * theta)
                # R = np.array(((c, -s), (s, c)))
                # pstart_rot = np.dot(R, pstart)
                # pend_rot = np.dot(R, pend)
                # rotated_normal = np.dot(R, normal)
                #

                # This work rotated around cntr !
                print("the radian angle is ", k * theta)
                pstart_rot = rotate_around_point(pstart, k * theta, (cntr[0], cntr[1]))
                pend_rot = rotate_around_point(pend, k * theta, (cntr[0], cntr[1]))
                # rotated_normal = rotate_around_point(normal, k * theta, (cntr[0], cntr[1]))
                # rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)

                ###
                # print("the radian angle is ", degrees(k * theta))
                # pstart_rot = rotate_around_point(pstart, k * theta, (0, 0))
                # pend_rot = rotate_around_point(pend, k * theta, (0, 0))
                rotated_normal = rotate_around_point(normal, k * theta, (0, 0))
                rotated_normal = rotated_normal / np.sqrt(rotated_normal[0] ** 2 + rotated_normal[1] ** 2)
                # print("the start",pstart_rot,"the end",pend_rot,"the normal",rotated_normal)
                edge_map_rgb = inp_img.copy()
                score_map, p1_max, value_max = Score_line(inp_img, rotated_normal,pstart_rot,pend_rot, x_len, edge_map_rgb)
                img_plot = score_map.copy()
                # print(p1_max)
                p1_max_arr[index_j, index_i, :] = p1_max
                vector_arr[index_j, index_i, :] = [-1 * rotated_normal[1], rotated_normal[0]]
                value_max_arr[index_j, index_i, :] = value_max
                degres_arr[index_j, index_i, :] = degrees(k * theta)

                index_i = index_i + 1

                # print(score_map.shape)
                #######################
                # Plot the result
                ######################


            ### Plot :D
            plt.figure()
            plt.plot(rangeofT, value_max_arr[index_j], 'ro')
            plt.axis(
                [np.min(rangeofT), np.max(rangeofT), np.min(value_max_arr[index_j]), np.max(value_max_arr[index_j])])
            plt.show()
            index_j = index_j + 1


        ######### Let calculate the result of both ############33

        value_max_arr_total = []
        plt.figure()

        for i in range(0, len(Input_images)):
            if (i == 0):
                value_max_arr_total = value_max_arr[i]
            else:
                value_max_arr_total = value_max_arr_total + value_max_arr[i]
        plt.plot(rangeofT, value_max_arr_total, 'ro')
        plt.axis([np.min(rangeofT), np.max(rangeofT), np.min(value_max_arr_total), np.max(value_max_arr_total)])
        plt.show()
        print(value_max_arr_total)
        score_total = value_max_arr_total
        max_point = np.argmax(score_total, axis=0)
        max_point = max_point[0]
        print("max_point", max_point)

        for i in range(0, len(Input_images)):
            if (i == 0):

                if (len(Input_images) == 1):
                    print("Only One Method ", p1_max_arr[0, max_point, :])
                mid_point = (p1_max_arr[0, max_point, :]).astype(int)
                mid_deg = degres_arr[0, max_point, :]

                print("The mid_point ", mid_point)

            else:
                print("Multiy Method ", p1_max_arr[0, max_point, :], p1_max_arr[1, max_point, :])
                mid_point = ((p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2).astype(int)
                print("The mid_point ", (p1_max_arr[0, max_point, :] + p1_max_arr[1, max_point, :]) // 2)

                print("Multiy Method ", degres_arr[0, max_point, :], degres_arr[1, max_point, :])
                mid_deg = ((degres_arr[0, max_point, :] + degres_arr[1, max_point, :]) // 2).astype(int)
                print("The mid_point ", (degres_arr[0, max_point, :] + degres_arr[1, max_point, :]) // 2)

        # max_normal = -normal_arr[0, max_point, :][0][0]

        # print(vector_arr)
        # print(degres_arr)
        # print(p1_max_arr)

        b = rotate_around_point(egenvec, np.radians(mid_deg), (0, 0))
        b = b / np.sqrt(b[0] ** 2 + b[1] ** 2)
        # b[0] = vector_arr[0, max_point, :][0]
        # b[1] = vector_arr[0, max_point, :][1]
        img_plot = img_ploty.copy()
        cv2.circle(img_plot, (mid_point[0], mid_point[1]), 5, (0, 0, 255), 1)
        cv2.imshow("the mid point", img_plot)

        print('The normals ', vector_arr[0, max_point, :], b)

        print(egenvectors)
        ps = (
        int(np.round(mid_point[0] + egenvec[0] * x_len / 2)), int(np.round(mid_point[1] + egenvec[1] * x_len / 2)))
        pe = (
        int(np.round(mid_point[0] - egenvec[0] * x_len / 2)), int(np.round(mid_point[1] - egenvec[1] * x_len / 2)))

        print(ps, pe)

        ps_2 = (int(np.round(mid_point[0] + b[0] * x_len / 2)),
                int(np.round(mid_point[1] + b[1] * x_len / 2)))
        pe_2 = (int(np.round(mid_point[0] - b[0] * x_len / 2)),
                int(np.round(mid_point[1] - b[1] * x_len / 2)))
        print(ps, pe)

        # img_plot = cv2.cvtColor(img_org.copy(), cv2.COLOR_GRAY2BGR)
        cv2.line(img_ploty, (ps[0], ps[1]), (pe[0], pe[1]), (255, 0, 0))
        cv2.line(img_ploty, (ps_2[0], ps_2[1]), (pe_2[0], pe_2[1]), (0, 0, 255))
        cv2.imshow("The re", img_ploty)
        cv2.waitKey(0)


    # print(final_lines)
    #
    # p1_in = final_lines[0][0]
    # p2_in = final_lines[0][1]
    # p3_in = final_lines[2][0]
    # p4_in = final_lines[2][1]
    #
    # p_inter1 = intersection2Lines(p1_in,p2_in,p3_in,p4_in)
    # print(p_inter1)
    #
    # p1_in = final_lines[0][0]
    # p2_in = final_lines[0][1]
    # p3_in = final_lines[3][0]
    # p4_in = final_lines[3][1]
    #
    # p_inter2 = intersection2Lines(p1_in,p2_in,p3_in,p4_in)
    # print(p_inter2)
    #
    # p1_in = final_lines[1][0]
    # p2_in = final_lines[1][1]
    # p3_in = final_lines[2][0]myplot
    # p4_in = final_lines[2][1]
    #
    # p_inter3 = intersection2Lines(p1_in, p2_in, p3_in, p4_in)
    # print(p_inter3)
    #
    # p1_in = final_lines[1][0]
    # p2_in = final_lines[1][1]
    # p3_in = final_lines[3][0]
    # p4_in = final_lines[3][1]
    #
    # p_inter4 = intersection2Lines(p1_in, p2_in, p3_in, p4_in)
    # print(p_inter4)
    #
    # img_plot2 = bw.copy()
    # for i in [p_inter1, p_inter2, p_inter3, p_inter4]:
    #     cv2.circle(img_plot2, (int(i[0]),int(i[1])), 3, (255,255  ,0), 2)
    #     cv2.imshow("The points",img_plot2)
    #     cv2.waitKey(0)
    #
    # corners = (np.array(np.array([p_inter1, p_inter2, p_inter3, p_inter4])))
    # corners = corners.reshape(4,2,1)
    # print(corners)


# Left and Right Snapping for  p2 and p4
    # We will need to look for x in axis world , or "Cols" in matrix world !!!!!!
    # pick the points that are closely for x


    # ####################################
    #  Left and Right
    ######################################


    # a = np.array(list_points)
    # ind = np.lexsort((a[:, 0], a[:, 1]))
    #
    # pstart = (a[ind][0][0],a[ind][0][1])
    # pend = (a[ind][-1][0],a[ind][-1][1])
    #
    # points = [pstart, pend ]
    # x_coords,y_coords = zip(*points)
    # print("The lines with  ", points)
    #
    # img_plot = imageTo90.copy()
    # cv2.line(img_plot, points[0], points[1], 255)
    # cv2.imshow("SnappingLine", img_plot)
    # cv2.waitKey()
    #
    # if points[0][0] == points[1][0] : # which mean the slope undefined !
    #     y_coordinates = np.arange(y_coords[0] , y_coords[1] + 1)
    #     x_coordinates = points[0][0] * np.ones(len(y_coordinates))
    #     print(x_coordinates)
    #     print(y_coordinates)
    #
    # else:
    #
    #     x_coords = np.sort(x_coords)
    #     print("x_coords",x_coords)
    #     A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    #     m, c = lstsq(A, y_coords)[0]
    #     print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    #     x_coordinates = np.arange(x_coords[0],x_coords[1]+1)
    #     y_coordinates = (m*x_coordinates+c).astype(int)
    #     print(x_coordinates)
    #     print(y_coordinates)
    #
    #
    #
    # cv2.destroyAllWindows()
    # plt.figure()
    # plt.imshow(img_plot)
    # plt.show()
    #
    # bw[bw == 0] = -255
    #
    # Left_p1,Left_p2 = SnapLine(bw,Direction.Left ,x_len,x_coordinates,y_coordinates,edge_map_rgb)
    # Right_p1, Right_p2 = SnapLine(bw,Direction.Right,x_len,x_coordinates,y_coordinates,edge_map_rgb)
    #
    #
    #
    # # ####################################
    # #  UP and Down
    # ######################################
    # a = np.array(list_points)
    # ind = np.lexsort((a[:, 1], a[:, 0]))
    #
    # pstart = (a[ind][0][0], a[ind][0][1])
    # pend = (a[ind][-1][0], a[ind][-1][1])
    # points = [pstart,pend]
    # x_coords, y_coords = zip(*points)
    # print("The lines with  ",points)
    #
    # img_plot = rotated_src.copy()
    # cv2.line(img_plot, points[0], points[1], 255)
    # cv2.imshow("SnappingLine",img_plot)
    # cv2.waitKey()
    #
    # if points[0][1] == points[0][0] : # which mean the slope undefined !
    #     y_coordinates = np.arange(y_coords[0] , y_coords[1] + 1)
    #     x_coordinates = points[0][0] * np.ones(len(y_coordinates))
    #
    # else:
    #
    #     x_coords = np.sort(x_coords)
    #     A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    #     m, c = lstsq(A, y_coords)[0]
    #     print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    #     x_coordinates = np.arange(x_coords[0],x_coords[1]+1)
    #     y_coordinates = (m*x_coordinates+c).astype(int)
    #     print(x_coordinates)
    #     print(y_coordinates)
    #
    # cv2.destroyAllWindows()
    # plt.figure()
    # plt.imshow(img_plot)
    # plt.show()
    # Up_p1,Up_p2=SnapLine(bw,Direction.Up,y_len,x_coordinates,y_coordinates,edge_map_rgb)
    # Down_p1,Down_p2 = SnapLine(bw,Direction.Down,y_len,x_coordinates,y_coordinates,edge_map_rgb)
    #
    #
    #
    # # PART 4
    # ##################################
    # # ets find the intersection  for all of them and Find Corners
    # ##################################
    #
    #
    # p1_left = (Left_p1[0][0], Left_p1[1])
    # p2_left = (Left_p1[0][-1], Left_p1[1])
    #
    # p1_leftxy = ( Left_p1[1],Left_p1[0][0])
    # p2_leftxy = (Left_p1[1],Left_p1[0][-1])
    #
    # p1_Up = (Up_p1[0], Up_p1[1][0])
    # p2_Up = (Up_p1[0], Up_p1[1][-1])
    #
    # p1_Upxy = (Up_p1[1][0],Up_p1[0])
    # p2_Upxy = (Up_p1[1][-1], Up_p1[0])
    #
    #
    # p1_right = (Right_p1[0][0], Right_p1[1])
    # p2_right = (Right_p1[0][-1], Right_p1[1])
    #
    # p1_rightxy = (Right_p1[1], Right_p1[0][0])
    # p2_rightxy = (Right_p1[1], Right_p1[0][-1])
    #
    #
    # p1_Down = (Down_p1[0], Down_p1[1][0])
    # p2_Down = (Down_p1[0], Down_p1[1][-1])
    #
    # p1_Downxy = (Down_p1[1][0],Down_p1[0])
    # p2_Downxy = (Down_p1[1][-1], Down_p1[0])
    #
    #
    # print("The points are left ",p1_left,p2_left)
    # print("The points are Down ", p1_Down, p2_Down)
    #
    #
    # p_interLeftUp = intersection2Lines(p1_leftxy,p2_leftxy,p1_Upxy,p2_Upxy)
    # print(p_interLeftUp)
    #
    # p_interRightUp = intersection2Lines(p1_rightxy, p2_rightxy, p1_Upxy, p2_Upxy)
    # print(p_interRightUp)
    #
    # p_interLeftDown = intersection2Lines(p1_rightxy, p2_rightxy,p1_Downxy, p2_Downxy)
    # print(p_interLeftDown)
    # p_interRightDown = intersection2Lines( p1_leftxy, p2_leftxy,p1_Downxy, p2_Downxy)
    # print(p_interRightDown)
    #
    #
    #
    # img_plot_sc = cv2.cvtColor(bw.copy(),cv2.COLOR_GRAY2RGB)
    # for i in [p_interLeftUp, p_interRightUp,p_interLeftDown,p_interRightDown]:
    #     cv2.circle(img_plot_sc, (int(i[0]),int(i[1])), 3, (255,0,0), 2)
    #     cv2.imshow("The points",img_plot_sc)
    #     cv2.waitKey(0)
    #
    # corners = (np.array(np.array([p_interLeftUp, p_interRightUp,p_interLeftDown,p_interRightDown])))
    # corners = corners.reshape(4,2,1)
    # print(corners)
    #

