import cv2
import numpy as np

def compute_stereo_geometry(keypoints_left, keypoints_right, good_matches, K):

    coordinates_left = np.float32([keypoints_left[i.queryIdx].pt for i in good_matches])
    coordinates_right = np.float32([keypoints_right[i.trainIdx].pt for i in good_matches])

    F, mask = cv2.findFundamentalMat(coordinates_left, coordinates_right, cv2.FM_RANSAC, 3.0, 0.99)

    inlier_matches = []
    for i, is_inlier in enumerate(mask):
        if is_inlier[0] == 1:
            inlier_matches.append(good_matches[i])

    E = K.T @ F @ K

    return F, E, inlier_matches
