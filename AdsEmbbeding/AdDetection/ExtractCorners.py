import cv2
import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage.segmentation import active_contour
from sklearn.neighbors import KDTree
from skimage.color import rgb2gray
from skimage.filters.rank import gradient
from skimage.filters import gaussian
from skimage.morphology import disk
import matplotlib.pyplot as plt
from sklearn import linear_model
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse
import sys
from itertools import cycle
from skimage import feature


def line_predict(params, X):
    m, c = params
    y = []

    for i in range(X.shape[0]):
        y.append(m * X[i] + c)

    return np.asarray(y)


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


def fit_lines(verts, img=None, SHOW_RESULTS=False):
    X = verts[:, 0].reshape(-1, 1)
    y = verts[:, 1].reshape(-1, 1)

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

        line_y_line = line_predict((m, c), line_X)
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

        points = np.zeros((line_X.shape[0], 2))

        points[:, 1] = line_X.reshape(-1)
        points[:, 0] = line_y_ransac.reshape(-1)

        m, c = find_line_model(points)
        m_lines.append((m, c))

        line_X = np.arange(X.min(), X.max())[:, np.newaxis]

        line_y_line = line_predict((m, c), line_X)
        if SHOW_RESULTS:
            ax.plot(line_X, line_y_line, color='red', linewidth=2)

        X = X[outlier_mask]
        y = y[outlier_mask]

    intersection_points = []

    for i in range(4):
        for j in range(i + 1, 4):
            if i == j:
                continue
            x = (m_lines[i][1] - m_lines[j][1]) / (m_lines[j][0] - m_lines[i][0])
            y = m_lines[i][0] * x + m_lines[i][1]
            if 0 < x < img.shape[0] and 0 < y < img.shape[1]:
                intersection_points.append((x, y))

    intersection_points = np.asarray(intersection_points)

    if SHOW_RESULTS:
        plt.scatter(intersection_points[:, 0], intersection_points[:, 1], color='yellow', marker='*', s=200)

        X = verts[:, 0].reshape(-1, 1)
        y = verts[:, 1].reshape(-1, 1)

        ax.set_xlim(X.min() - 50, X.max() + 50)
        ax.set_ylim(y.min() - 50, y.max() + 50)
        plt.show()

    return intersection_points


def get_four_points_polygon(cnt, w=1000, h=1000, SHOW_RESULTS=False):
    cnt = np.asarray(cnt, dtype=np.float32)
    rect = cv2.minAreaRect(cnt)
    rect = cv2.boxPoints(rect)

    src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    dst_pts = rect.reshape((-1, 2))

    H, mask = cv2.findHomography(dst_pts, src_pts)

    dst = cv2.perspectiveTransform(cnt.reshape((-1, 1, 2)), H).reshape((-1, 2))

    if SHOW_RESULTS:
        I = np.ones((w, h)) * 0
        print(dst.shape[0])
        for j in range(dst.shape[0]):
            print((dst[j][0], dst[j][1]))
            I = cv2.circle(I, (int(dst[j][0]), int(dst[j][1])), 2, 100, 3)

    kdt = KDTree(dst, leaf_size=30, metric='euclidean')

    indices = kdt.query(np.asarray([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]), k=1, return_distance=False)

    verts = [cnt[indices[j][0]] for j in range(4)]

    if SHOW_RESULTS:
        for j in range(4):
            I = cv2.circle(I, (int(dst[indices[j][0]][0]), int(dst[indices[j][0]][1])), 20, 100, 3)
        cv2.imshow('projected', I)
        cv2.waitKey(0)

    return np.asarray(verts)


def get_line_normal(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)

    reg = linear_model.LinearRegression().fit(X, y)
    n = np.asarray([reg.coef_[0][0], 1])
    n = n / np.linalg.norm(n)
    n = np.asarray([-1 * n[1], n[0]])

    return n


def extract_peaks(contours, M=20, thresh=0.7):
    peaks = []

    N = len(contours)

    for i in range(N):
        before = []
        after = []
        for j in range(M):
            before.append(contours[(i - M + j) % N])
            after.append(contours[(i + j) % N])

        before = np.asarray(before)
        after = np.asarray(after)

        b_n = get_line_normal(before)
        a_n = get_line_normal(after)
        if 0.7 > np.dot(b_n, a_n) > 0.5:
            peaks.append(contours[i])

    return np.asarray(peaks)


def add_vertices(img, verts, distance=20, epsilon=0, SHOW_RESULTS=False):
    if SHOW_RESULTS:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.scatter(verts[:, 0], verts[:, 1], color='yellowgreen', marker='.')

    n_verts = []
    v = 0
    while v < verts.shape[0]:

        p1 = verts[v]
        p2 = verts[(v + 1) % verts.shape[0]]

        dist = np.linalg.norm(p1 - p2)

        p = verts[v]

        if dist < distance:
            p = (p1 + p2) / 2
            if SHOW_RESULTS:
                ax.scatter(np.asarray([p[0]]), np.asarray([p[1]]), color='green', marker='.')

        n_verts.append(p)

        v += 1

    i = 0
    while i < len(n_verts):
        n = len(n_verts)
        p1 = n_verts[i]
        p2 = n_verts[(i + 1) % n]

        dist = np.linalg.norm(p1 - p2)

        if dist > distance + epsilon:
            n_p = (p1 + p2) / 2
            n_verts.insert(i + 1, n_p)
            i -= 1
            if SHOW_RESULTS:
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

    if SHOW_RESULTS:
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()

    return np.asarray(n_verts)


def draw_keypoints(vis, keypoints, color=(0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 2, color, -1)


def detect_image_corners(img, SHOW_RESULTS=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    harris_img = corner_harris(gray, method='eps')
    if SHOW_RESULTS:
        plt.imshow(harris_img)
        plt.show()

    return harris_img


def detect_image_edgaes(img, T=2, SHOW_RESULTS=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = np.zeros_like(img)
    for i in range(T):
        edges += feature.canny(img, sigma=1 + 2 * i)

    if SHOW_RESULTS:
        plt.imshow(edges)
        plt.show()

    # grad = gradient(img, disk(3))
    # plt.imshow(grad)
    # plt.show()
    #
    # nr, nc = edges.shape
    # dx, dy = np.gradient(edges)
    # x, y = np.meshgrid(np.arange(1,nc),np.arange(1,nr))
    # u = dx
    # v = dy
    # plt.quiver(x, y, u, v)
    # plt.show()

    return edges


def active_contours(img, verts, SHOW_RESULTS=False):
    img = rgb2gray(img.copy())
    # verts = np.fliplr(verts)

    snake = active_contour(gaussian(img, 3),
                           verts, bc='periodic', alpha=0.001, beta=0.001, gamma=0.01, w_edge=200)
    if SHOW_RESULTS:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(verts[:, 0], verts[:, 1], '--r', lw=1)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()

    return snake


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def morphological_snakes(img, verts, SHOW_RESULTS=False):
    from skimage import data, img_as_float
    from skimage.segmentation import (morphological_chan_vese,
                                      morphological_geodesic_active_contour,
                                      inverse_gaussian_gradient,
                                      checkerboard_level_set)
    img = rgb2gray(img.copy())
    image = img_as_float(img)

    # Initial level set
    init_ls = checkerboard_level_set(img.shape, 6)

    # verts = np.array(verts).reshape((-1,1,2)).astype(np.int32)
    # init_ls = np.zeros(image.shape, dtype=np.int8)
    # cv2.drawContours(init_ls, [verts], -1, 1, 2)

    plt.imshow(init_ls)
    plt.show()
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 32, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 32")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)

    return ls


def extract_lines(img, T=4, SHOW_RESULTS=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = np.zeros_like(gray)
    for i in range(T):
        edges += feature.canny(gray, sigma=1 + 2 * i)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if SHOW_RESULTS:
        plt.imshow(edges)
        plt.show()

    return lines


def test(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    edged = cv2.Canny(gray, 10, 250)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closed", closed)
    cv2.waitKey(0)

    _, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the approximated contour has four points, then assume that the
        # contour is a book -- a book is a rectangle and thus has four vertices
        if len(approx) == 4:
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
            total += 1

    # display the output
    print("I found {0} books in that image".format(total))
    cv2.imshow("Output", img)
    cv2.waitKey(0)


def refine_boundaries(img, boundary, step=10, M=3, SHOW_RESULTS=False):
    refined_boundary = boundary
    N = len(boundary)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    d_img = img.copy()

    lines_img = np.zeros((img.shape[0], img.shape[1], 3))
    lsd = cv2.createLineSegmentDetector(0)
    # Detect lines in the image
    dlines = lsd.detect(gray)  # Position 0 of the returned tuple are the detected lines

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        X = [x0, x1]
        y = [y0, y1]

        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        reg = linear_model.LinearRegression().fit(X, y)

        n = np.asarray([reg.coef_[0][0], 1])
        n = n / np.linalg.norm(n)

        n = np.asarray([-1 * n[0], n[1]])

        cv2.line(lines_img, (x0, y0), (x1, y1), (n[0], n[1], 0), 4)

    if SHOW_RESULTS:
        plt.imshow(lines_img[:, :, 0])
        plt.show()
        plt.imshow(lines_img[:, :, 1])
        plt.show()

    vertices = []
    boundaries_img = np.zeros((img.shape[0], img.shape[1], 3))

    for i in range(N):
        p1 = boundary[i]

        X = []
        y = []
        for k in range(i - M, i + M):
            k = k % N
            X.append(boundary[k][0])
            y.append(boundary[k][1])
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        reg = linear_model.LinearRegression().fit(X, y)

        n = np.asarray([reg.coef_[0][0], 1])
        n = n / np.linalg.norm(n)

        n = np.asarray([-1 * n[0], n[1]])

        cv2.circle(boundaries_img, (int(p1[0]), int(p1[1])), 2, (n[0], n[1], 0), -1)

        max_point = p1.copy()
        if int(p1[0]) >= lines_img.shape[0] or int(p1[0]) < 0 \
                or int(p1[1]) >= lines_img.shape[1] or int(p1[1]) < 0:
            continue
        max_ori = 0
        for j in range(-step, step, 1):
            p = p1 + j * n
            if np.isnan(p[0]) or np.isnan(p[1]) or int(p[0]) >= lines_img.shape[0] or int(p[0]) < 0 \
                    or int(p[1]) >= lines_img.shape[1] or int(p[1]) < 0:
                continue

            cv2.circle(lines_img, (int(p[0]), int(p[1])), 1, (0, 0, 255))
            l_n = lines_img[int(round(p[0])), int(round(p[1]))]
            l_n = np.asarray((l_n[0], l_n[1]))
            ori = np.linalg.norm(n - l_n)
            ori = np.dot(l_n, n)
            print('p: {}, t: {}, ori: {}'.format(n, l_n, ori), file=sys.stderr)
            if ori > max_ori:
                print('max: {}'.format(ori), file=sys.stderr)
                max_point = p
                max_ori = ori

        if SHOW_RESULTS:
            cv2.line(d_img, (int(p1[0] - step * n[0]), int(p1[1] - step * n[1])),
                     (int(p1[0] + step * n[0]), int(p1[1] + step * n[1])), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(d_img, (int(max_point[0]), int(max_point[1])), 2, (0, 0, 255), -1)
            cv2.circle(d_img, (int(p1[0]), int(p1[1])), 2, (0, 255, 0), -1)
            cv2.imshow('lines_img', d_img)
            cv2.waitKey(1)

        refined_boundary[i] = max_point
        #
        # if min_ori < 0.5:
        #     vertices.append(max_point)

    plt.imshow(boundaries_img[:, :, 0])
    plt.show()
    plt.imshow(boundaries_img[:, :, 1])
    plt.show()

    return refined_boundary


def getSpPoint(A, B, C):
    x1 = A[0]
    y1 = A[1]
    x2 = B[0]
    y2 = B[1]
    x3 = C[0]
    y3 = C[1]
    px = x2 - x1
    py = y2 - y1
    dAB = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
    x = x1 + u * px
    y = y1 + u * py
    return np.asarray([x, y])


def is_point_on_line(p1, p2, q):
    if (p1[0] == p2[0]) and (p1[1] == p2[1]):
        p1[0] -= 0.00001

    U = ((q[0] - p1[0]) * (p2[0] - p1[0])) + ((q[1] - p1[1]) * (p2[1] - p1[1]))
    Udenom = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
    U /= Udenom

    r = [0, 0]
    r[0] = p1[0] + (U * (p2[0] - p1[0]))
    r[1] = p1[1] + (U * (p2[1] - p1[1]))

    minx = min(p1[0], p2[0])
    maxx = max(p1[0], p2[0])
    miny = min(p1[1], p2[1])
    maxy = max(p1[1], p2[1])

    is_valid = (minx <= r[0] <= maxx) and (miny <= r[1] <= maxy)

    if is_valid:
        return r
    else:
        return None


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


def minimum_distance(v, w, p):
    proj = is_point_on_line(v, w, p)

    if proj is None:
        return min(np.linalg.norm(v - p), np.linalg.norm(w - p))
    return np.linalg.norm(proj - p)


def snap_to_edges(img, boundary, thresh=100, SHOW_RESULTS=False):
    N = len(boundary)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    d_img = img.copy()

    lsd = cv2.createLineSegmentDetector(0)
    # Detect lines in the image
    dlines = lsd.detect(gray)  # Position 0 of the returned tuple are the detected lines

    for i in range(N):
        p = boundary[i]

        min_point = p
        min_dist = np.inf

        for dline in dlines[0]:

            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))

            for t_p in [[x0, y0], [x1, y1]]:
                t_p = np.asarray(t_p)
                dist = np.linalg.norm(t_p - p)
                if dist < thresh and dist < min_dist:
                    min_point = t_p
                    min_dist = dist

        boundary[i] = min_point

        if SHOW_RESULTS:
            cv2.line(d_img, (int(min_point[0]), int(min_point[1])), (int(p[0]), int(p[1])), (255, 0, 0), 1)
            cv2.circle(d_img, (int(min_point[0]), int(min_point[1])), 2, (0, 0, 255), -1)
            cv2.circle(d_img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
            cv2.imshow('corner snap', d_img)
            cv2.waitKey(1)

    return boundary


def refine_boundaries2(img, boundary, alpha=0.1, M=10, dist_thresh=50, SHOW_RESULTS=False):
    refined_boundary = boundary
    N = len(boundary)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    d_img = img.copy()

    lines_img = np.zeros((img.shape[0], img.shape[1], 3))
    lsd = cv2.createLineSegmentDetector(0)
    # Detect lines in the image
    dlines = lsd.detect(gray)  # Position 0 of the returned tuple are the detected lines

    for i in range(N):
        p1 = boundary[i]

        X = []
        y = []
        for k in range(i - M, i + M):
            k = k % N
            X.append(boundary[k][0])
            y.append(boundary[k][1])
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        p_reg = linear_model.LinearRegression().fit(X, y)

        n = np.asarray([p_reg.coef_[0][0], 1])
        n = n / np.linalg.norm(n)

        n = np.asarray([-1 * n[0], n[1]])

        max_point = p1.copy()
        min_score = img.shape[0]
        for dline in dlines[0]:

            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))

            cv2.line(d_img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

            X = [x0, x1]
            y = [y0, y1]

            X = np.asarray(X).reshape(-1, 1)
            y = np.asarray(y).reshape(-1, 1)

            l_reg = linear_model.LinearRegression().fit(X, y)

            l_n = np.asarray([l_reg.coef_[0][0], 1])
            l_n = l_n / np.linalg.norm(l_n)

            l_n = np.asarray([-1 * l_n[0], l_n[1]])

            dist = minimum_distance(np.asarray([x0, y0]), np.asarray([x1, y1]), p1)

            proj_point = getSpPoint([x0, y0], [x1, y1], p1)

            # print('[{}-{}], proj:{} ,dist: {}'.format([x0, y0], [x1, y1], proj_point, dist), file=sys.stderr)

            ori = np.dot(l_n, n)

            score = alpha * (1 - ori) + (1 - alpha) * dist

            # print('[{}->{}] n: {}, l_n:{} , score: {}, dist: {}, ori: {}'.format(p1, proj_point, n, l_n, score, dist,
            #                                                                      ori), file=sys.stderr)
            if score < min_score and dist < dist_thresh:
                # print('max: {}'.format(ori), file=sys.stderr)
                max_point = proj_point
                min_score = score

        if SHOW_RESULTS:
            cv2.imwrite('lines_img.jpg', d_img)
            cv2.line(d_img, (int(max_point[0]), int(max_point[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 1)
            cv2.circle(d_img, (int(max_point[0]), int(max_point[1])), 2, (0, 0, 255), -1)
            cv2.circle(d_img, (int(p1[0]), int(p1[1])), 2, (0, 255, 0), -1)
        refined_boundary[i] = max_point

    if SHOW_RESULTS:
        cv2.imshow('lines_img', d_img)
        cv2.waitKey(1)

    return refined_boundary


def fit_lines2(points, img, SHOW_RESULTS=False):
    lines = []

    fit_points = points.copy()

    d_img = img.copy()

    for i in range(4):
        line = cv2.fitLine(points, distType=cv2.DIST_HUBER, param=0, reps=0.01, aeps=0.01)
        lines.append(line)

        vx = line[0]
        vy = line[1]
        x = line[2]
        y = line[3]

        lefty = np.round((-x * vy / vx) + y)
        righty = np.round(((img.shape[1] - x) * vy / vx) + y)
        point1 = np.asarray([img.shape[1] - 1, righty])
        point2 = np.asarray([0, lefty])

        t = []
        for j in range(len(fit_points)):
            dist = minimum_distance(point1, point2, fit_points[j])

            if dist > 50:
                t.append(fit_points[j])

    for line in lines:
        vx = line[0]
        vy = line[1]
        x = line[2]
        y = line[3]

        cv2.line(d_img, (int(x - vx * img.shape[0]), int(y - vy * img.shape[0])),
                 (int(x + vx * img.shape[0]), int(y + vy * img.shape[0])), (255, 0, 0), 1)
        cv2.circle(d_img, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imshow('flines', d_img)
    cv2.waitKey(0)


def fit_lines3(points, img, SHOW_RESULTS=False):
    contour = np.asarray(points.copy(), dtype=np.float32).reshape(-1, 2)

    area = cv2.contourArea(contour)

    # moments:重心
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    # 重心を中心にした円
    cv2.circle(img, (cx, cy), int(np.sqrt(area)), (0, 255, 0), 1, 8)

    # 最小外接円
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    img = cv2.circle(img, center, radius, (255, 255, 0), 2)

    # 輪郭の近似
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # print(len(approx), file=sys.stderr)
    if len(approx) == 4:
        cv2.drawContours(img, [approx.astype(int)], -1, (255, 0, 0), 3)

    # 凸包
    hull = cv2.convexHull(contour)
    img = cv2.drawContours(img, [hull.astype(int)], 0, (255, 255, 0), 2)

    # 外接矩形
    rect = cv2.boundingRect(contour)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)

    # 回転を考慮した外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img, [box.astype(int)], 0, (0, 255, 255), 1)

    # 楕円のフィッティング
    ellipse = cv2.fitEllipse(contour)
    img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    # 直線のフィッティング
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    cv2.imshow('findContours', img)
    cv2.imwrite('findContours.png', img)

    cv2.waitKey(0)


def calculate_weights(contours):
    weights = {}
    for i in range(len(contours)):
        tri = [contours[i - 1], contours[i], contours[(i + 1) % len(contours)]]
        tri = np.asarray(tri, dtype=np.float32)

        area = cv2.contourArea(tri)
        weights[i] = area

    return weights


def relax_poly(contours, img=None, DRAW_RESULTS=False):
    if DRAW_RESULTS and img is not None:
        d_img = img.copy()

    contours = contours.tolist()
    weights = calculate_weights(contours)

    while len(contours) > 4:
        min_p = min(weights, key=weights.get)

        if DRAW_RESULTS and img is not None:
            cv2.circle(d_img, (int(contours[min_p][0]), int(contours[min_p][1])), 2, (255, 0, 0), -1)

        contours.pop(min_p)
        weights = calculate_weights(contours)

    if DRAW_RESULTS and img is not None:
        for p in contours:
            cv2.circle(d_img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        cv2.imshow('relaxed contours', d_img)
        cv2.waitKey(0)
        cv2.imwrite('relaxed_img.png', d_img)

    return np.asarray(contours, dtype=np.float32)


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


    for n_l in best_line:
        vx = n_l[0]
        vy = n_l[1]

        if np.abs(vx) < 1e-05:
            vx = 0.0001
        if np.abs(vy) < 1e-05:
            vy = 0.0001

        lefty = int((-n_l[2] * vy / vx) + n_l[3])
        righty = int(((img.shape[1] - n_l[2]) * vy / vx) + n_l[3])
        cv2.line(d_img, (img.shape[1] - 1, righty), (0, lefty), (255, 0, 0), 1)

    intersection_points = []

    for i in range(4):
        for j in range(i + 1, 4):
            if i == j:
                continue

            x = np.asarray([best_line[j][2], best_line[j][3]]) - np.asarray([best_line[i][2], best_line[i][3]])

            cross = best_line[i][0] * best_line[j][1] - best_line[i][1] * best_line[j][0]
            if abs(cross) < 1e-8:
                return False

            t1 = (x[0] * best_line[j][1] - x[1] * best_line[j][0]) / cross
            r = np.asarray([best_line[i][2], best_line[i][3]]) + np.asarray([best_line[i][0], best_line[i][1]]) * t1

            if 0 < r[0] < img.shape[1] and 0 < r[1] < img.shape[0]:
                cv2.circle(d_img, (int(r[0]), int(r[1])), 4, (0, 0, 255), -1)
                intersection_points.append((r[0], r[1]))

    # print('ER: {}'.format(intersection_points), file=sys.stderr)

    def getKey(ele):
        return ele[0]

    intersection_points.sort(key=getKey)

    intersection_points = np.asarray(intersection_points, dtype=np.float32)
    # intersection_points = np.sort(intersection_points.view('i8,i8'), order=['f0', 'f1'], axis=0).view(np.float32)

    # print('sorted ER: {}'.format(intersection_points), file=sys.stderr)

    # cv2.imshow('lines', d_img)
    cv2.imwrite('rotated.png', d_img)
    # cv2.waitKey(0)

    return intersection_points


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def order_4_points(points, width, height):
    from scipy.optimize import linear_sum_assignment

    corners = np.array([[height, width], [0, height], [0,0], [width, 0]]).reshape(-1,2)
    cost = np.zeros((4,4))

    for i in range(4):
        for j in range(4):
            cost[i, j] = distance(points[i], corners[j])

    row_ind, col_ind = linear_sum_assignment(cost)


    print('col: {}, row: {}'.format(col_ind, row_ind))

    ordered_points = [points[col_ind[0]], points[col_ind[1]],
                      points[col_ind[2]], points[col_ind[3]]]

    return np.asarray(ordered_points)

def extract_corners(img, masks, min_area=100):
    img_gray = rgb2gray(img)
    gradient_image = gradient(img_gray, disk(1))
    img_gray = np.asarray(img_gray, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of instances
    N = masks.shape[2]
    if not N:
        print("\n*** No instances to display *** \n")

    refined_contours = []

    for i in range(N):

        mask = masks[:, :, i]
        contours = find_contours(mask, 0.5)
        for verts in contours:

            try:
                _contours = np.asarray(verts.copy(), dtype=np.float32).reshape(-1, 2)
                rect = cv2.minAreaRect(_contours)
                box = cv2.boxPoints(rect)

                box_area = cv2.contourArea(box)

                c_area = cv2.contourArea(_contours)

                if c_area/box_area < 0.7:
                    continue

                verts = np.fliplr(verts)

                # verts = approximate_polygon(verts, 0.0001)

                verts = refine_boundaries2(img, verts, SHOW_RESULTS=True)

                # fit_lines3(verts, img=img, SHOW_RESULTS=True)

                # verts = add_vertices(img, verts, SHOW_RESULTS=True)
                #
                # verts = fit_lines(verts, img=img, SHOW_RESULTS=True)

                # verts = snap_to_edges(img, verts)

                # verts = extract_peaks(verts)

                # verts = approximate_polygon(verts, 0.01)

                # lsd = cv2.createLineSegmentDetector(0)

                # Detect lines in the image
                # lines = lsd.detect(gray)[0]  # Position 0 of the returned tuple are the detected lines

                # Draw detected lines in the image
                # drawn_img = lsd.drawSegments(gray, lines)

                # for line in lines:
                #     x0 = int(round(line[0][0]))
                #     y0 = int(round(line[0][1]))
                #     x1 = int(round(line[0][2]))
                #     y1 = int(round(line[0][3]))
                #     cv2.circle(img, (x0, y0), 2, (255, 0, 0), 4)
                #     cv2.circle(img, (x1, y1), 2, (255, 0, 0), 4)
                #
                # # Show image
                # cv2.imshow('ad', img)
                # cv2.waitKey(1)
                #
                # plt.imshow(drawn_img)
                # plt.show()

                # morphological_snakes(img, verts, True)
                #
                # verts = active_contours(img, verts, SHOW_RESULTS=True)
                #
                # plt.figure(0)
                # plt.imshow(img)
                # plt.scatter(verts[:, 0], verts[:, 1])
                # plt.show()
                #
                # # verts = peaks
                #
                # verts = add_vertices(img, verts, SHOW_RESULTS=True)
                #
                #
                # verts = approximate_polygon(verts, 50)
                #
                # plt.figure(0)
                # plt.imshow(img)
                # plt.scatter(verts[:, 0], verts[:, 1])
                # plt.show()

                verts = extract_4_poly(verts, img, T=10, M=10)

                if verts.shape[0] < 4:
                    print('Less than 4 points', file=sys.stderr)
                    continue


                verts = get_four_points_polygon(verts)


                # verts = order_4_points(verts, img.shape[1], img.shape[0])

                area = cv2.contourArea(verts)

                if area > min_area:
                    refined_contours.append(verts)

            except Exception as e:
                print('Something went wrong when refining ad_region contours', file=sys.stderr)
                print('Error: {}'.format(e), file=sys.stderr)

    return refined_contours
