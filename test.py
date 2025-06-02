import cv2
import numpy as np

# Extract matrices and vectors from the .npz file
calibration_data = np.load('Y:/Priyanshu Davda/python/stereo_calibration.npz')

K1 = calibration_data['mtx1.npy']  # Intrinsic matrix for the left camera
K2 = calibration_data['mtx2.npy']  # Intrinsic matrix for the right camera
dist1 = calibration_data['dist1.npy']  # Distortion coefficients for the left camera
dist2 = calibration_data['dist2.npy']  # Distortion coefficients for the right camera
R = calibration_data['R.npy']  # Rotation matrix
T = calibration_data['T.npy']  # Translation vector

# Load rectified stereo images
left_image = cv2.imread('rectified_left_image.jpg')
right_image = cv2.imread('rectified_right_image.jpg')

# Convert images to grayscale
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(left_gray, None)
kp2, des2 = orb.detectAndCompute(right_gray, None)

# Use BFMatcher to find matches
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(left_image, None)
kp2, des2 = orb.detectAndCompute(right_image, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)


# Sort matches based on distance
matches = sorted(matches, key = lambda x:x.distance) 

# Draw matches
img_matches = cv2.drawMatches(left_image, kp1, right_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Select corresponding points
points_left = np.array([kp1[m.queryIdx].pt for m in matches[:10]], dtype=np.float32)
points_right = np.array([kp2[m.trainIdx].pt for m in matches[:10]], dtype=np.float32)

# Compute disparity map using StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,
    blockSize=11,
    P1=8 * 3 * 11**2,
    P2=32 * 3 * 11**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_HH
)


disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# Display disparity map
cv2.imshow('Disparity Map', disparity / disparity.max())
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate 3D coordinates for each point
focal_length = K1[0, 0]  # Assuming both cameras have the same focal length
baseline = np.linalg.norm(T)

for pt_left, pt_right in zip(points_left, points_right):
    disparity_value = pt_left[0] - pt_right[0]
    if disparity_value > 0:
        depth = (focal_length * baseline) / disparity_value
        x = (pt_left[0] - K1[0, 2]) * depth / focal_length
        y = (pt_left[1] - K1[1, 2]) * depth / focal_length
        print(f"3D Coordinates: X={x}, Y={y}, Z={depth}")
    else:
        print("Invalid disparity value.")
