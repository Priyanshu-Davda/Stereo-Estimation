import cv2
import numpy as np
import glob

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real world space)
# Assume chessboard pattern is 5x8 squares, each square having size 30mm
# You can change this depending on your actual chessboard dimensions
chessboard_size = (5, 8)  # Number of inner corners in the chessboard
square_size = 30  # Size of a square in millimeters

# 3D points in real world space (0,0,0), (1,0,0), (2,0,0), ..., (5,7,0)
obj_points = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
obj_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
obj_points *= square_size  # Scale by the square size

# Arrays to store object points and image points
obj_points_list = []  # 3D points in world space
img_points_list = []  # 2D points in image plane

# Load the images of the chessboard pattern
# Ensure the path below contains the calibration images
image_paths = glob.glob('left_frames_old/*.jpg')  # Change this path to your image folder

frame_counter = 0  # Initialize a frame counter

# Process each image and collect points
for idx, image_path in enumerate(image_paths):
    # Check if the frame number is a multiple of 15
    if idx % 15 != 0:
        continue  # Skip all frames that are not every 15th frame

    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        obj_points_list.append(obj_points)  # Add 3D points for calibration
        # Refine the corner locations to subpixel accuracy
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points_list.append(corners)  # Add 2D points for calibration
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)  # Wait for a key press before proceeding

# Close the window once all images are processed
cv2.destroyAllWindows()

# Perform camera calibration to obtain camera matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, img_points_list, gray.shape[::-1], None, None,flags=cv2.CALIB_RATIONAL_MODEL)

# Print the calibration results
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

# Save the results (optional)
np.save('mtx.npy', mtx)
np.save('dist.npy', dist)

# Undistort one of the images using the calibration parameters
img = cv2.imread(image_paths[50])
dst = cv2.undistort(img, mtx, dist)
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
