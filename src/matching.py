import cv2

def detect_and_match_features(img_left, img_right):
    """
    Detects keypoints in both images and finds high-quality matching pairs.
    """
    orb = cv2.ORB_create(nfeatures=1000)

    keypoints_left, descriptors_left = orb.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)

    bruteforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    initial_matches = bruteforce.knnMatch(descriptors_left, descriptors_right, k=2)

    good_matches = []
    for m, n in initial_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return keypoints_left, keypoints_right, good_matches