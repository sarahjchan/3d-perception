import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as _helper

def compute_fundamental_matrix(pts1, pts2, scale):
    """
    Compute the Fundamental matrix from corresponding 2D points in two images.

    Given two sets of corresponding 2D image points from Image 1 (pts1) and Image 2 (pts2),
    as well as a scaling factor (scale) representing the maximum dimension of the images, 
    this function calculates the Fundamental matrix.

    Parameters:
    pts1 (numpy.ndarray): An Nx2 array containing 2D points from Image 1.
    pts2 (numpy.ndarray): An Nx2 array containing 2D points from Image 2, corresponding to pts1.
    scale (float): The maximum dimension of the images, used for scaling the Fundamental matrix.

    Returns:
    F (numpy.ndarray): A 3x3 Fundamental matrix 
    """
    F = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    T = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
    pts1_normalized = pts1 * (1/scale)
    pts2_normalized = pts2 * (1/scale)
    N = np.shape(pts1)[0]
    A = np.zeros((N, 9))

    for i in range(N):
        pt1 = pts1_normalized[i]
        pt2 = pts2_normalized[i]

        x1 = pt1[0]
        x2 = pt2[0]
        y1 = pt1[1]
        y2 = pt2[1]

        A[i, :] = np.array([[x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]])

    U, S, Vt = np.linalg.svd(A)

    f = Vt.T[:, -1]
    f = np.reshape(f, (3, 3))

    U_F, S_F, Vt_F = np.linalg.svd(f)
    S_F[2] = 0

    F = U_F @ np.diag(S_F) @ Vt_F
    F = T.T @ F @ T
    # _helper.epipolar_lines_GUI_tool(img1, img2, F)
    ####################################
    return F

def compute_epipolar_correspondences(img1, img2, pts1, F):
    """
    Compute epipolar correspondences in Image 2 for a set of points in Image 1 using the Fundamental matrix.

    Given two images (img1 and img2), a set of 2D points (pts1) in Image 1, and the Fundamental matrix (F)
    that relates the two images, this function calculates the corresponding 2D points (pts2) in Image 2.
    The computed pts2 are the epipolar correspondences for the input pts1.

    Parameters:
    img1 (numpy.ndarray): The first image containing the points in pts1.
    img2 (numpy.ndarray): The second image for which epipolar correspondences will be computed.
    pts1 (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates img1 and img2.

    Returns:
    pts2_ep (numpy.ndarray): An Nx2 array of corresponding 2D points (pts2) in Image 2, serving as epipolar correspondences
                   to the points in Image 1 (pts1).
    """
    pts2_ep = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    N = np.shape(pts1)[0]
    pts2_ep = np.zeros((N, 2))
    ones = np.ones((N, 1))
    pts1 = np.hstack((pts1, ones))
    best_distance = 9999999
    height = np.shape(img1)[0]
    width = np.shape(img1)[1]

    for i in range(N):
        pt1 = pts1[i]
        line = F @ pt1
        pt_x = pt1[0]
        pt_y = pt1[1]

        if pt_x <= width - 2 and pt_y <= height - 2 and pt_x >= 2 and pt_y >= 2:
            pt2_xs = range(width)
            pt2_ys = -(line[0]*pt2_xs + np.ones_like(pt2_xs)*line[2])/line[1]
            for x in range(2, width - 1):
                pt2_x = x
                pt2_y = int(pt2_ys[x])
                window_pt1 = img1[int(pt_y)-2:int(pt_y)+2, int(pt_x)-2:int(pt_x)+2]
                window_pt2 = img2[int(pt2_y)-2:int(pt2_y)+2, int(pt2_x)-2:int(pt2_x)+2]
                distance = np.sum(window_pt1 - window_pt2)
                if distance < best_distance:
                    best_distance = distance
                    best_x = x

        best_y = int(pt2_ys[best_x])
        pts2_ep[i] = np.array([best_x, best_y])
    ####################################
    return pts2_ep

def compute_essential_matrix(K1, K2, F):
    """
    Compute the Essential matrix from the intrinsic matrices and the Fundamental matrix.

    Given the intrinsic matrices of two cameras (K1 and K2) and the 3x3 Fundamental matrix (F) that relates
    the two camera views, this function calculates the Essential matrix (E).

    Parameters:
    K1 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 1.
    K2 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 2.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates Camera 1 and Camera 2.

    Returns:
    E (numpy.ndarray): The 3x3 Essential matrix (E) that encodes the essential geometric relationship between
                   the two cameras.

    """
    E = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    E = np.linalg.inv(K2.T) @ F @ np.linalg.inv(K1)
    ####################################
    return E 

def triangulate_points(E, pts1_ep, pts2_ep):
    """
    Triangulate 3D points from the Essential matrix and corresponding 2D points in two images.

    Given the Essential matrix (E) that encodes the essential geometric relationship between two cameras,
    a set of 2D points (pts1_ep) in Image 1, and their corresponding epipolar correspondences in Image 2
    (pts2_ep), this function calculates the 3D coordinates of the corresponding 3D points using triangulation.

    Extrinsic matrix for camera1 is assumed to be Identity. 
    Extrinsic matrix for camera2 can be found by cv2.decomposeEssentialMat(). Note that it returns 2 Rotation and 
    one Translation matrix that can form 4 extrinsic matrices. Choose the one with the most number of points in front of 
    the camera.

    Parameters:
    E (numpy.ndarray): The 3x3 Essential matrix that relates two camera views.
    pts1_ep (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    pts2_ep (numpy.ndarray): An Nx2 array of 2D points in Image 2, corresponding to pts1_ep.

    Returns:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point.
    point_cloud_cv (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point calculated using cv2.triangulate
    """
    point_cloud = None
    point_cloud_cv = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    N = np.shape(pts1_ep)[0]
    point_cloud = np.zeros((N, 3))
    ones = np.ones((N, 1))
    pts1_ep = np.hstack((pts1_ep, ones))
    pts2_ep = np.hstack((pts2_ep, ones))

    extrinsic1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    R1, R2, t = cv2.decomposeEssentialMat(E)
    extrinsic2 = np.hstack((-R2, t))
    print('E', E)
    print('R1', R1)
    print('R2', R2)

    x1 = pts1_ep[:, 0]
    y1 = pts1_ep[:, 1]

    x2 = pts2_ep[:, 0]
    y2 = pts2_ep[:, 1]

    p11 = extrinsic1[0, :].T
    p21 = extrinsic1[1, :].T
    p31 = extrinsic1[2, :].T
    
    p12 = extrinsic2[0, :].T
    p22 = extrinsic2[1, :].T
    p32 = extrinsic2[2, :].T
    

    for i in range(N):
        print('pt1', x1[i], y1[i])
        print('pt2', x2[i], y2[i])
        A = np.array([[y1[i]*p31 - p21], [p11 - x1[i]*p31], [y2[i]*p32 - p22], [p12 - x2[i]*p32]])

        U, S, Vt = np.linalg.svd(A)
        V = Vt.T
        pt = V[:, -1]
        pt = (1/pt[-1])*pt
        print('pt', pt)
        point_cloud[i] = pt[:3]
        print('pt', pt)

    # point_cloud_cv = cv2.triangulatePoints(extrinsic1, extrinsic2, pts1_ep, pts2_ep)
    print('pc shape', np.shape(point_cloud))
    ####################################
    return point_cloud, point_cloud_cv


def visualize(point_cloud):
    """
    Function to visualize 3D point clouds
    Parameters:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud,where each row contains the 3D coordinates
                   of a triangulated point.
    """
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()
    ####################################


if __name__ == "__main__":
    data_for_fundamental_matrix = np.load("data/corresp_subset.npz")
    pts1_for_fundamental_matrix = data_for_fundamental_matrix['pts1']
    pts2_for_fundamental_matrix = data_for_fundamental_matrix['pts2']

    img1 = cv2.imread('data/im1.png')
    img2 = cv2.imread('data/im2.png')
    scale = max(img1.shape)
    

    data_for_temple = np.load("data/temple_coords.npz")
    pts1_epipolar = data_for_temple['pts1']

    data_for_intrinsics = np.load("data/intrinsics.npz")
    K1 = data_for_intrinsics['K1']
    K2 = data_for_intrinsics['K2']

    ####################################
    ##########YOUR CODE HERE############
    ####################################
    F = compute_fundamental_matrix(pts1_for_fundamental_matrix, pts2_for_fundamental_matrix, scale)
    pts2_epipolar = compute_epipolar_correspondences(img1, img2, pts1_epipolar, F)
    # _helper.epipolar_correspondences_GUI_tool(img1, img2, F)
    E = compute_essential_matrix(K1, K2, F)
    point_cloud, point_cloud_cv = triangulate_points(E, pts1_epipolar, pts2_epipolar)
    visualize(point_cloud)
    ####################################




