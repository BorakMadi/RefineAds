import os

import tensorflow as tf
import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.filters.rank import gradient
from skimage.io import imread
import json
import cv2
from sklearn.neighbors import KDTree
from sklearn import linear_model
import sys
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize

from ExtractCorners import extract_corners
from ExtractCorners2 import extract_corners2

import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()

SHOW_RESULTS = False
SHOW_COMP_RESULTS = False

#######################import datetime

now = datetime.datetime.now()#####################################
#  Configurations
############################################################
class AdRegionsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "Ad-Regions"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + Ad regions

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def refine_boundaries(gradients_image, boundary, step=5, M=8):
    refined_boundary = boundary
    N = len(boundary)

    d_gradients_image = gradients_image.copy()

    for i in range(N):
        p0 = boundary[i - 1]
        p1 = boundary[i]
        p2 = boundary[(i + 1) % N]

        X = []
        y = []
        for k in range(i-M, i+M):
            k = k % N
            X.append(boundary[k][0])
            y.append(boundary[k][1])
        X = np.asarray(X).reshape(-1,1)
        y = np.asarray(y).reshape(-1,1)

        reg = linear_model.LinearRegression().fit(X, y)

        # n = np.asarray([reg.predict(np.asarray([1]).reshape(-1,1))[0] - reg.predict(np.asarray([0]).reshape(-1,1))[0], 1])

        n = np.asarray([reg.coef_[0][0], 1])
        n = n / np.linalg.norm(n)

        n = np.asarray([-1 * n[1], n[0]])

        # print('{}, {}, {} - {}'.format(p0, p1, p2, n))

        max_point = p1.copy()
        if int(p1[0]) >= gradients_image.shape[0] or int(p1[0]) < 0 \
                or int(p1[1]) >= gradients_image.shape[1] or int(p1[1]) < 0:
            continue
        max_grad = gradients_image[int(p1[0]), int(p1[1])]
        for j in range(-step, step, 1):
            p = p1 + j * n
            if np.isnan(p[0]) or np.isnan(p[1]) or int(p[0]) >= gradients_image.shape[0] or int(p[0]) < 0 \
                    or int(p[1]) >= gradients_image.shape[1] or int(p[1]) < 0:
                continue

            cv2.circle(gradients_image, (int(p[0]), int(p[1])), 1, (0, 0, 255))
            grad = gradients_image[int(p[0]), int(p[1])]
            if grad > max_grad:
                max_point = p
                max_grad = grad

        # print('{} -> {}'.format(p1, max_point))
        if SHOW_RESULTS:
            cv2.circle(d_gradients_image, (int(max_point[0]), int(max_point[1])), 5, (255, 0, 0), 4)
            cv2.circle(d_gradients_image, (int(p1[0]), int(p1[1])), 3, (0, 255, 0), 4)
            cv2.imshow('grad', d_gradients_image)
            cv2.waitKey(1)

        refined_boundary[i] = max_point

    # ax.imshow(gradients_image), plt.show()

    # while len(refined_boundary) > 4:
    #     num_points = len(refined_boundary)
    #
    #     max_anv = 0
    #     max_i = 0
    #     for i in range(num_points):
    #         p0 = boundary[i - 1]
    #         p1 = boundary[i]
    #         p2 = boundary[(i + 1) % N]
    #
    #         n1 = p2 - p1
    #         n2 = p0 - p1
    #
    #         an_v = np.dot(n1, n2)
    #
    #         if max_anv < an_v:
    #             max_anv = an_v
    #             max_i = i
    #
    #     refined_boundary = np.delete(refined_boundary, max_i, 0)

    return refined_boundary


def get_four_points_polygon(cnt, w=1000, h=1000):
    cnt = np.asarray(cnt, dtype=np.float32)
    rect = cv2.minAreaRect(cnt)
    rect = cv2.boxPoints(rect)

    src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    dst_pts = rect.reshape((-1, 2))

    H, mask = cv2.findHomography(dst_pts, src_pts)

    dst = cv2.perspectiveTransform(cnt.reshape((-1, 1, 2)), H).reshape((-1, 2))

    # I = np.ones((w,h))*0
    # print(dst.shape[0])
    # for j in range(dst.shape[0]):
    #     print((dst[j][0],dst[j][1]))
    #     I = cv2.circle(I, (int(dst[j][0]),int(dst[j][1])), 2, 100, 3)

    kdt = KDTree(dst, leaf_size=30, metric='euclidean')

    indices = kdt.query(np.asarray([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]), k=1, return_distance=False)

    verts = [cnt[indices[j][0]] for j in range(4)]

    # for j in range(4):
    #     I = cv2.circle(I, (int(dst[indices[j][0]][0]), int(dst[indices[j][0]][1])), 20, 100, 3)
    # cv2.imshow('projected', I)
    # cv2.waitKey(0)

    return np.asarray(verts)


def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """

    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero

    # find a line model for these points
    m = (points[1, 1] - points[0, 1]) / (
                points[1, 0] - points[0, 0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1, 1] - m * points[1, 0]  # y-intercept of the line

    return m, c


def high_grad_points(verts, M=4):
    N = len(verts)
    h_g_points = []
    for i in range(N):
        X = []
        y = []
        for k in range(i - M, i+1):
            k = k % N
            X.append(verts[k][0])
            y.append(verts[k][1])
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        reg = linear_model.LinearRegression().fit(X, y)

        p_n = np.asarray([reg.coef_[0][0], 1])
        p_n = p_n / np.linalg.norm(p_n)

        p_n = np.asarray([-1 * p_n[1], p_n[0]])

        X = []
        y = []
        for k in range(i, i + M+1):
            k = k % N
            X.append(verts[k][0])
            y.append(verts[k][1])
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        reg = linear_model.LinearRegression().fit(X, y)

        n_n = np.asarray([reg.coef_[0][0], 1])
        n_n = n_n / np.linalg.norm(n_n)

        n_n = np.asarray([-1 * n_n[1], n_n[0]])

        grad = np.linalg.norm(p_n - n_n)
        if grad > 1:
            h_g_points.append(verts[i])
            plt.scatter(verts[0], verts[1], color='red', marker='*')
            # plt.text(verts[0] , verts[1] , '{}'.format(grad), fontsize=9)

    h_g_points = np.asarray(h_g_points)

    if SHOW_RESULTS:
        plt.show()

    return h_g_points


def hough_liens(img, verts):
    img2 = img.copy()
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 3, np.pi / 180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if SHOW_RESULTS:
        cv2.imshow('canny', edges)
        cv2.imshow('linens', img2)
        cv2.waitKey(0)

    return verts


def line_predict(params, X):
    m, c = params
    y = []

    for i in range(X.shape[0]):
        y.append(m*X[i] + c)

    return np.asarray(y)
def fit_lines(verts, img=None):

    X = verts[:,0].reshape(-1,1)
    y = verts[:, 1].reshape(-1,1)

    # plt.scatter(X, y, color='red', marker='.',
    #             label='Inliers')
    lines = []

    if SHOW_RESULTS:
        fig, ax = plt.subplots(figsize=(10, 10))
        if img is not None:
            plt.imshow(img)

    m_lines = []
    for _ in range(2):
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor(residual_threshold=20)
        ransac.fit(X, y)

        lines.append(ransac)

        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        if SHOW_RESULTS:
            ax.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=3,
                     label='RANSAC regressor')
            ax.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                        label='Inliers')

            ax.scatter(X[outlier_mask], y[outlier_mask], color='red', marker='.', label='Outliers')


        points = np.zeros((line_X.shape[0], 2))

        points[:, 0] = line_X.reshape(-1)
        points[:, 1] = line_y_ransac.reshape(-1)

        m, c = find_line_model(points)
        m_lines.append((m, c))

        line_y_line = line_predict((m,c), line_X)
        if SHOW_RESULTS:
            ax.plot(line_X, line_y_line, color='red', linewidth=1)

        # plt.show()
        X = X[outlier_mask]
        y = y[outlier_mask]

    for _ in range(2):
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor(residual_threshold=20)
        ransac.fit(y, X)

        lines.append(ransac)

        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # Predict data of estimated models
        line_X = np.arange(y.min(), y.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        # ax.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=3,
        #          label='RANSAC regressor')
        # ax.scatter(y[inlier_mask], X[inlier_mask], color='yellowgreen', marker='.',
        #             label='Inliers')
        #
        # ax.scatter(y[outlier_mask], X[outlier_mask], color='red', marker='.', label='Outliers')

        points = np.zeros((line_X.shape[0], 2))

        points[:, 1] = line_X.reshape(-1)
        points[:, 0] = line_y_ransac.reshape(-1)

        m, c = find_line_model(points)
        m_lines.append((m, c))

        line_X = np.arange(X.min(), X.max())[:, np.newaxis]

        line_y_line = line_predict((m,c), line_X)
        if SHOW_RESULTS:
            ax.plot(line_X, line_y_line, color='red', linewidth=2)

        X = X[outlier_mask]
        y = y[outlier_mask]

    # X = verts[:, 0].reshape(-1, 1)
    # y = verts[:, 1].reshape(-1, 1)
    #
    # ax.set_xlim(X.min()-50, X.max()+50)
    # ax.set_ylim(y.min()-50, y.max()+50)
    # plt.show()

    intersection_points = []

    for i in range(4):
        for j in range(i+1,4):
            if i == j:
                continue
            x = (m_lines[i][1] - m_lines[j][1])/(m_lines[j][0] - m_lines[i][0])
            y = m_lines[i][0] * x + m_lines[i][1]
            intersection_points.append((x,y))

    intersection_points = np.asarray(intersection_points)

    if SHOW_RESULTS:
        # fig, ax = plt.subplots(figsize=(10, 10))
        plt.scatter(intersection_points[:, 0], intersection_points[:, 1], color='yellow', marker='*', s=200)

        X = verts[:, 0].reshape(-1, 1)
        y = verts[:, 1].reshape(-1, 1)

        ax.set_xlim(X.min()-50, X.max()+50)
        ax.set_ylim(y.min()-50, y.max()+50)
        plt.show()

    return intersection_points


def active_contours(img, verts):

    img = rgb2gray(img.copy())
    # verts = np.fliplr(verts)


    snake = active_contour(gaussian(img, 1),
                           verts, bc='periodic', alpha=0.001, beta=100, gamma=10, w_edge=500000)
    if SHOW_RESULTS:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(verts[:, 0], verts[:, 1], '--r', lw=1)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()

    return snake


def add_vertices(img, verts, distance=20, epsilon=0):

    if SHOW_RESULTS:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.scatter(verts[:, 0], verts[:, 1], color='yellowgreen', marker='.')


    n_verts = []
    v = 0
    while v < verts.shape[0]:

        p1 = verts[v]
        p2 = verts[(v+1)%verts.shape[0]]

        dist = np.linalg.norm(p1-p2)

        p = verts[v]

        if dist < distance:
            p = (p1+p2)/2
            ax.scatter(np.asarray([p[0]]), np.asarray([p[1]]), color='green', marker='.')

        n_verts.append(p)

        v += 1

    i = 0
    while i < len(n_verts):
        n = len(n_verts)
        p1 = n_verts[i]
        p2 = n_verts[(i+1)%n]

        dist = np.linalg.norm(p1-p2)

        if dist > distance + epsilon:
            n_p = (p1+p2)/2
            n_verts.insert(i+1, n_p)
            i -= 1
            ax.scatter(np.asarray([n_p[0]]), np.asarray([n_p[1]]), color='green', marker='.')
        # elif dist < distance - epsilon:
        #     n_p = (p1 + p2) / 2
        #
        #     n_verts.pop(i)
        #
        #     n_verts.insert(i-1, n_p)
        #     ax.scatter(np.asarray([p1[0]]), np.asarray([p1[1]]), color='red', marker='.')
        #     ax.scatter(np.asarray([p2[0]]), np.asarray([p2[1]]), color='red', marker='.')
        #
        #     ax.scatter(np.asarray([n_p[0]]), np.asarray([n_p[1]]), color='yellow', marker='.')
        #     i -= 1
        i += 1

    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

    return np.asarray(n_verts)


def masks_boundaries2(img,testname, masks,folderName=None):
    plt.ion()
    N = masks.shape[2]
    if not N:
        print("\n*** No instances to display *** \n")

    # Let's compute PCA eigenVectors

    for i in range(N):
        mask = masks[:, :, i]
        plt.figure()
        plt.imshow(mask)
        plt.show()

        if folderName is not None:
            corners = extract_corners2(img,testname,mask.copy(),folderName)
        else:
            corners = extract_corners2(img, testname, mask.copy())

    return corners



def masks_boundaries(img, masks, min_area=100):

    img_gray = rgb2gray(img)
    gradient_image = gradient(img_gray, disk(1))
    if SHOW_RESULTS:
        plt.imshow(gradient_image)
        plt.show()
    # Number of instances
    N = masks.shape[2]
    if not N:
        print("\n*** No instances to display *** \n")

    refined_contours = []

    for i in range(N):

        mask = masks[:, :, i]

        if SHOW_RESULTS:
            plt.imshow(mask)
            plt.show()


        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(mask, 0.5)
        if SHOW_RESULTS:
            plt.imshow(padded_mask)
            plt.show()

        for verts in contours:

            verts = np.fliplr(verts)
            verts = approximate_polygon(verts, 0.01)
            verts = refine_boundaries(gradient_image, verts)

            # verts = approximate_polygon(verts, 50)
            # verts = cv2.minAreaRect(np.asarray(verts, dtype=np.float32))
            # verts = cv2.boxPoints(verts)
            # verts = active_contours(img, verts)
            # verts = approximate_polygon(verts, 1)
            #
            # verts = add_vertices(img, verts)
            #
            # verts = fit_lines(verts, img=img)
            #
            # corners_points = []
            #
            # for i in range(verts.shape[0]):
            #     if img.shape[0] > verts[i, 0] >= 0 and img.shape[1] > verts[i, 1] >= 0:
            #         corners_points.append(verts[i])
            # verts = np.asarray(corners_points)
            #
            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.imshow(img, cmap=plt.cm.gray)
            # ax.scatter(verts[:, 0], verts[:, 1], color='blue', marker='*', s=200)
            # plt.show()
            # verts = hough_liens(img, verts)
            verts = get_four_points_polygon(verts)
            area = cv2.contourArea(verts)

            if area > min_area:
                refined_contours.append(verts)

    return refined_contours


config = AdRegionsConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

M = 10


class AdDetection:
    def __init__(self):
        with tf.device(DEVICE):
            self.model = modellib.MaskRCNN(mode="inference", model_dir="",
                                           config=config)

    def load_weights(self, weights):
        # print("Loading weights ", weights)
        self.model.load_weights(weights, by_name=True)

    def detect_ad(self, image):
        results = self.model.detect([image], verbose=1)
        boundaries = masks_boundaries2(image, results[0]['masks'])


        # _, ax = plt.subplots(1, 1, figsize=(16, 16))
        # for j in range(len(boundaries)):
        #     p = Polygon(boundaries[j], facecolor="none", edgecolor=(0, 1, 0, 0.8))
        #     ax.add_patch(p)
        # plt.imshow(image.astype(np.uint8))
        # plt.draw()

        return boundaries


import sys

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Detect Ads boundaries.')
    parser.add_argument('--image', required=False,
                        metavar="/path/to/image",
                        help='image to apply the detection method on')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument("--size", required=True,
                        default="420x360", help="resolution of background image (e.g. 420x360)")
    # parser.add_argument('--readFrames', required=True,
    #                     default=1,
    #                     help="Number of frames to read from the pipe")
    args = parser.parse_args()

    # print("Weights: ", args.weights)

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir="",
                                  config=config)
    weights_path = args.weights
    H, W = [int(x) for x in args.size.split("x")]
    # NFrames = int(args.readFrames)

    # Load weights
    # print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


    ###################### our New Test ##########################333
    import glob
    print(" We are in Main")
    test_cases = []
    test_folder='Images'
    files = glob.glob("{}/*".format(test_folder))
    for f in files:
        if '.png' in f or '.jpg' in f or '.bmp' in f:
            name = f.split('.')[0]
            name = name.split('/')[-1]
            test_cases.append((f, name))

    name_folder = 'Outputs/'+now.strftime("%H:%M:%S")+'/'
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)

    for testfile, testname in test_cases:
    #
        image = cv2.imread(testfile)
        results = model.detect([image], verbose=0)

        boundaries = masks_boundaries2(image,testname,results[0]['masks'],name_folder)
        # res = {"image": args.image, "boundaries": [b.tolist() for b in boundaries]}
        # print(json.dumps(res, ensure_ascii=False), '\n')




    # if args.image is not None:
    #
    #     image = imread(args.image)
    #     # Run detection
    #     # print("We are Here")
    #     results = model.detect([image], verbose=0)
    #     boundaries = masks_boundaries2(image, results[0]['masks'])
    #     res = {"image": args.imasks_boundaries2mage, "boundaries": [b.tolist() for b in boundaries]}
    #     print(json.dumps(res, ensure_ascii=False), '\n')
    #
    #
    # else:
    #     while True:
    #
    #         # read W*H*3 bytes (= NFrames frame)
    #         raw_image = sys.stdin.buffer.read(W * H * 3)
    #         print('input image: {}'.format(raw_image), file=sys.stderr)
    #         # transform the byte read into a numpy array
    #         image = np.fromstring(raw_image, dtype='uint8')
    #         image = image.reshape((H, W, 3))
    #
    #         # throw away the data in the pipe's buffer.
    #
    #         results = model.detect([image], verbose=1)
    #         # ax = plt.figure(1)
    #         # r = results[0]
    #         # if SHOW_RESULTS:
    #         #     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #         #                                 'ad', r['scores'],
    #         #                                 title="Predictions")
    #
    #         boundaries = masks_boundaries2(image, results[0]['masks'])
    #         # if SHOW_RESULTS:
    #         #     for b in boundaries:
    #         #         plt.imshow(image)
    #         #         plt.scatter(b[:, 0], b[:,1])
    #         #     plt.show()
    #
    #         res = {"boundaries": [b.tolist() for b in boundaries]}
    #         print("res : ", json.dumps(res, ensure_ascii=False), file=sys.stderr)
    #         sys.stdout.write(json.dumps(res, ensure_ascii=False))
    #         sys.stdout.write("\n")
    #         print('{}'.format(json.dumps(res, ensure_ascii=False)))
