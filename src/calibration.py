import numpy as np

def parse_kitti_calib(filepath):
    """
    Parses a KITTI calibration text file and extracts the projection matrices
    for the left and right color cameras.

    Args:
        filepath (str): Path to the calibration text file (e.g., 'calib.txt')

    Returns:
        tuple: (P_left, P_right) as 3x4 NumPy arrays.
    """
    calib_dict = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue

            key, value = line.split(':', 1)
            # Convert the space-separated string of numbers into a flat NumPy array
            calib_dict[key.strip()] = np.fromstring(value.strip(), sep=' ')

    if 'P2' not in calib_dict or 'P3' not in calib_dict:
        raise KeyError("Could not find color camera projection matrices P2 or P3 in the calibration file.")

    P_left = calib_dict['P2'].reshape(3, 4)
    P_right = calib_dict['P3'].reshape(3, 4)
    return P_left, P_right

def extract_camera_parameters(P):
    """
    Extracts the intrinsic matrix (K) from a 3x4 projection matrix P.
    For a rectified camera, P = [K | K * t], where K is the 3x3 intrinsic matrix.
    """
    K = P[:, :3]
    return K

def calculate_baseline(P_left, P_right):
    """
        Calculates the physical baseline distance between the left and right cameras.
        The projection matrix P = [K | T], where T_x = f_x * t_x.
        Baseline is the absolute difference between physical translations (t_x).

        Args:
            P_left (numpy.ndarray): 3x4 projection matrix of the left camera.
            P_right (numpy.ndarray): 3x4 projection matrix of the right camera.

        Returns:
            float: The baseline distance in meters.
    """
    T_x_left = P_left[0,3]
    T_x_right = P_right[0,3]
    f_x = P_left[0, 0]
    baseline = abs(T_x_left - T_x_right) / f_x
    return float(baseline)