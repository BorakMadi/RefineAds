from __future__ import print_function
import cv2 as cv2
import numpy as np
import argparse
import random as rng
from itertools import product
import matplotlib.pyplot as plt
rng.seed(12345)

def find_corners(pts):
    print(pts.reshape(-1,2))
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


def main():


    ##### MASK !
    mask = cv2.imread("/home/borak/Results_Snapping/Snapping Point/mask-res.png")
    # img = DistanceMap(mask)
    print(mask.shape)
    mask_gray = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    _, mask_gray_bw = cv2.threshold(mask_gray, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(mask_gray_bw )
    cv2.imshow("mask_gray_bw", mask_gray_bw)
    cv2.waitKey(0)
    mask_gray_bw_cor = mask_gray_bw.copy()


    src = cv2.imread("/home/borak/Results_Snapping/Snapping Point/im_res_scl.png")

    #################
    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # so the possible negative number will be truncated
    imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
    sharp = np.float32(src)

    for i in [0]:
        imgResult = sharp - imgLaplacian
        imgLaplacian = cv2.filter2D(imgResult, cv2.CV_32F, kernel)

    imgResult = cv2.cvtColor(imgResult,cv2.COLOR_RGB2GRAY)
    cv2.imshow("imgResult2",imgResult)
    cv2.waitKey(0)


    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    cv2.imshow('Laplace Filtered Image', imgLaplacian)
    cv2.imshow('New Sharped Image', 255-imgResult)
    cv2.waitKey(0)

    inv_img_Results =  255-imgResult
    resized = cv2.resize(inv_img_Results, (mask_gray_bw.shape[1], mask_gray_bw.shape[0]), interpolation=cv2.INTER_AREA)




    ############
    # Extened Contour to hit the edges
    _, contours, _ = cv2.findContours(mask_gray_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c);
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue
        corners, x_len, y_len = find_corners(c)

        # for x in corners:
        #     cv2.circle(rot90, x, 3, (255, 255, 255), 2)
        # cv2.imshow("unrot", rot90)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        break

    print(corners)
    corners_arry = np.array(corners)
    ind = np.lexsort((corners_arry[:, 0], corners_arry[:, 1 ]))
    tmp = ind[-1]
    ind[-1]=ind[-2]
    ind[-2]=tmp
    print(ind)
    print(corners_arry[ind])
    extende_mask = cv2.fillConvexPoly(np.zeros_like(mask_gray_bw), corners_arry[ind],255)
    cv2.imshow("extende_mask", extende_mask)
    cv2.waitKey(0)
    alpha = 3
    z = np.multiply(resized, 255-extende_mask) + alpha * extende_mask
    cv2.imshow("inverse", z)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    z = cv2.cvtColor(z,cv2.COLOR_GRAY2RGB)
    distance_grad_img = DistanceMap(z)
    distance_mask = DistanceMap(mask)

    cv2.destroyAllWindows()
    cv2.imshow("distance_grad_img", distance_grad_img)
    cv2.imshow("distance_mask", distance_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(combine)
    # print(z.shape)
    # cv2.imshow("combine", z)
    # cv2.waitKey(0)
    # src[mask != 0] = (0, 0, 255)
    #
    #



    # src = cv2.imread("/home/borak/Results_Snapping/Snapping Point/im_res_scl.png")

def DistanceMap(src):

    if src is None:
        print('Could not open or find the image:')
        exit(0)
    # Show source image

    #######################
    #
    #  Loading Images and  "Try" to zero the background if it's white.!
    #
    #################

    cv2.imshow('Source Image', src)
    print(src.shape)
    src[np.all(src == 255, axis=2)] = 0
    # Show output image
    cv2.imshow('Black Background Image', src)
    cv2.waitKey()
    cv2.destroyAllWindows()


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
    imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
    sharp = np.float32(src)

    imgResult = sharp - imgLaplacian
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    cv2.imshow('Laplace Filtered Image', imgLaplacian)
    cv2.imshow('New Sharped Image', imgResult)
    cv2.waitKey()
    cv2.destroyAllWindows()


    #######################
    #
    #  Third Stage : we will sharpen calc distance Map
    #
    #################

    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('Binary Image', bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow('Distance Transform Image', dist)
    cv2.waitKey(0)

    _, dist = cv2.threshold(dist, 0.8, 1.0, cv2.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1)
    cv2.imshow('Peaks', dist)

    dist_8u = dist.astype('uint8')

    cv2.destroyAllWindows()

    # Find total markers
    _, contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)
    # Draw the background marker
    cv2.circle(markers, (5,5), 3, (255,255,255), -1)
    cv2.imshow('Markers', markers*10000)
    cv2.watershed(imgResult, markers)
    #mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv2.bitwise_not(mark)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    #cv.imshow('Markers_v2', mark)
    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]
    # Visualize the final image
    cv2.imshow('Final Result', dst)
    cv2.imshow('Black Background Image', src)
    cv2.waitKey()

    cv2.destroyAllWindows()
    return dist
if __name__ == '__main__':
    main()