import numpy as np
import cv2


def calculate_GCF(img, ddepth=-1, scale=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gcf = np.zeros((img.shape[0], img.shape[1], 2))

    grad_x = cv2.Scharr(img, ddepth, 1, 0, scale)
    grad_y = cv2.Scharr(img, ddepth, 0, 1, scale)

    gcf[:, :, 0] = grad_x
    gcf[:, :, 1] = grad_y

    return gcf


def CPC(region, T=10):
    curr_frame = region.original_frame
    gcf = calculate_GCF(curr_frame)

    tmp_img = np.zeros_like((curr_frame.shape[0], curr_frame.shape[1], 1))
    for j in range(4):
        cv2.line(tmp_img, region.region_corners[j % 4], region.region_corners[(j + 1) % 4], 255 - j * 50, thickness=1)

    for i in range(T):
        for j in range(4):
            vpj = 0.5 * (np.sum(gcf[tmp_img == (255 - j * 50)]) / np.count_nonzero(tmp_img == (255 - j * 50)))
            print(vpj)
            region.region_corners[j] += vpj


def CPC_2k_frames(first_frame_index, regions, k=10):
    for i in range(first_frame_index, first_frame_index + 2 * k):
        CPC(regions[i])
