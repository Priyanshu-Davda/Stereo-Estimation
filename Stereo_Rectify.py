import cv2
import numpy as np

# Load the stereo images (left and right images)
left_image = cv2.imread("left_frames/frame_0300.jpg")  # Use the path to your left image
right_image = cv2.imread("right_frames/frame_0300.jpg")  # Use the path to your right image

# Load the stereo calibration parameters (you should already have these from the stereo calibration)
# Camera 1 Matrix (intrinsic parameters)
calibration_data = np.load('stereo_calibration.npz')

# Extract matrices and vectors from the .npz file
K1 = calibration_data['mtx1.npy']  # Intrinsic matrix for the left camera
K2 = calibration_data['mtx2.npy']  # Intrinsic matrix for the right camera
D1 = calibration_data['dist1.npy']  # Distortion coefficients for the left camera
D2 = calibration_data['dist2.npy']  # Distortion coefficients for the right camera
R = calibration_data['R.npy']  # Rotation matrix
T = calibration_data['T.npy']  # Translation vector

# Image size (width, height)
h, w = left_image.shape[:2]

# Compute the rectification transformation matrices
# This step uses stereoRectify to compute the rectification transformation and projection matrices
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T)

# Compute the undistortion and rectification maps
map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32F)
map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32F)

# Rectify the images using the map
rectified_left_image = cv2.remap(left_image, map1_x, map1_y, cv2.INTER_LINEAR)
rectified_right_image = cv2.remap(right_image, map2_x, map2_y, cv2.INTER_LINEAR)

# Display the rectified images
cv2.imshow('Rectified Left Image', rectified_left_image)
cv2.imshow('Rectified Right Image', rectified_right_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the rectified images if needed
cv2.imwrite('rectified_left_image.jpg', rectified_left_image)
cv2.imwrite('rectified_right_image.jpg', rectified_right_image)
