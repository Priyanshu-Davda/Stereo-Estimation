import cv2
import numpy as np

# Load the stereo calibration data from .npz file
calibration_data = np.load('stereo_calibration.npz')

# Extract matrices and vectors from the .npz file
K1 = calibration_data['mtx1.npy']  # Intrinsic matrix for the left camera
K2 = calibration_data['mtx2.npy']  # Intrinsic matrix for the right camera
dist1 = calibration_data['dist1.npy']  # Distortion coefficients for the left camera
dist2 = calibration_data['dist2.npy']  # Distortion coefficients for the right camera
R = calibration_data['R.npy']  # Rotation matrix
T = calibration_data['T.npy']  # Translation vector

# Calculate the baseline distance (B) using the norm of the translation vector
baseline = np.linalg.norm(T)

# Load the stereo rectified images
left_rectified_image = cv2.imread('rectified_left_image.jpg')  # Path to your rectified left image
right_rectified_image = cv2.imread('rectified_right_image.jpg')  # Path to your rectified right image

# Initialize clicked points
clicked_point_left = None
clicked_point_right = None

# Function to handle mouse click to get the point for both images
def mouse_callback_left(event, x, y, flags, param):
    global clicked_point_left
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point_left = (x, y)
        cv2.circle(left_rectified_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Left Rectified Image", left_rectified_image)

def mouse_callback_right(event, x, y, flags, param):
    global clicked_point_right
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point_right = (x, y)
        cv2.circle(right_rectified_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Right Rectified Image", right_rectified_image)

# Show the left rectified image and wait for a mouse click
cv2.imshow("Left Rectified Image", left_rectified_image)
cv2.setMouseCallback("Left Rectified Image", mouse_callback_left)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()

# Show the right rectified image and wait for a mouse click
cv2.imshow("Right Rectified Image", right_rectified_image)
cv2.setMouseCallback("Right Rectified Image", mouse_callback_right)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()

# Check if both points are clicked
if clicked_point_left and clicked_point_right:
    print(f"Left Image Clicked Point: {clicked_point_left}")
    print(f"Right Image Clicked Point: {clicked_point_right}")

    # Get the pixel coordinates from both images
    x_left, y_left = clicked_point_left
    x_right, y_right = clicked_point_right

    # Convert the left and right rectified images to grayscale
    left_gray = cv2.cvtColor(left_rectified_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified_image, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's StereoBM for disparity calculation (assuming rectified images)
    # Example for StereoSGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Must be divisible by 16
        blockSize=11,       # Odd number, usually between 3 and 11
        P1=8 * 3 * 11**2,   # 8 * channels * blockSize^2
        P2=32 * 3 * 11**2,  # 32 * channels * blockSize^2
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode = cv2.STEREO_SGBM_MODE_HH
        # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # Or STEREO_SGBM_MODE_HH for more accuracy
)

    # Compute the disparity map
    disparity_map = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # Normalize disparity values

    # Display the disparity map for debugging (optional)
    cv2.imshow('Disparity Map', disparity_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate the disparity for the selected points
    disparity = disparity_map[y_left, x_left]
    print(f"Disparity at point ({x_left}, {y_left}): {disparity}")

    # Ensure the disparity is non-negative
    if disparity > 0:
        # Calculate the 3D coordinates using the stereo camera parameters
        # Formula: Z = (f * B) / disparity
        # Focal length in pixels from the camera matrix
        f1 = K1[0, 0]  # From Camera 1 intrinsic matrix

        # Calculate 3D coordinates in camera 1 frame
        Z = (f1 * baseline) / disparity  # Z coordinate (depth)
        X = (x_left - K1[0, 2]) * Z / f1  # X coordinate in real-world space
        Y = (y_left - K1[1, 2]) * Z / f1  # Y coordinate in real-world space

        print(f"3D Coordinates (X, Y, Z): {X}, {Y}, {Z}")
    else:
        print("Invalid disparity value (negative or zero). Skipping 3D calculation.")
else:
    print("Error: You need to click on corresponding points in both images.")
