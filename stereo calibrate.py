import cv2
import numpy as np
import glob
import math

# Define checkerboard dimensions

leftmatrix = np.array([[-938.776, 0, 304.406], [0, 940.118, 263.606], [0.0, 0.0, 1.0]])
rightmatrix = np.array([[1001.087, 0, 278.863], [0, 1004.272, 236.639], [0.0, 0.0, 1.0]])
leftDist = np.array([ -0.287,
0.927,
-0,
0.001,
-0.984])
rightDist = np.array([-0.466,
6.312,
0.005,
0.012,
-40.4820])
checkerboard_rows = 5  # Number of inner corners per checkerboard row
checkerboard_cols = 8  # Number of inner corners per checkerboard column
square_size = 30  # Size of each square in millimeters

# Prepare object points based on real-world coordinates (3D points)
objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
objp *= square_size  # Convert to millimeters

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints1 = []  # 2D points in the first camera's image plane
imgpoints2 = []  # 2D points in the second camera's image plane

# Load the frames from the extracted images for both cameras
left_images = sorted(glob.glob('left_frames/*.jpg'))  # Adjust the path if needed
right_images = sorted(glob.glob('right_frames/*.jpg'))  # Adjust the path if needed
# left_images = sorted(glob.glob('left_frames_old/*.jpg'))  # Adjust the path if needed
# right_images = sorted(glob.glob('right_frames_old/*.jpg'))  # Adjust the path if needed

# Check if image pairs exist
if len(left_images) == 0 or len(right_images) == 0:
    print("No images found in left_frames or right_frames directory. Please check the file paths.")
else:
    print(f"Found {len(left_images)} left images and {len(right_images)} right images.")

# Ensure there is a 1-to-1 correspondence between left and right images
valid_image_pairs = 0

# Use every 5th image
TotalFrames = len(left_images)
MaxFrames = 45
step = math.ceil(TotalFrames/MaxFrames)

for i in range(0, len(left_images), step):
    if i < len(right_images):  # Make sure we don't go out of bounds
        left_img = left_images[i]
        right_img = right_images[i]

        # Read the images
        img1 = cv2.imread(left_img)
        img2 = cv2.imread(right_img)

        if img1 is None or img2 is None:
            print(f"Error loading images: {left_img}, {right_img}")
            continue

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners in both images
        ret1, corners1 = cv2.findChessboardCorners(gray1, (checkerboard_cols, checkerboard_rows), None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (checkerboard_cols, checkerboard_rows), None)

        # Debugging: Print if corners are found
        if ret1 and ret2:
            print(f"Checkerboard found in {left_img} and {right_img}")
            
            # Refine the corner locations to subpixel accuracy
            cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))  # FIX: Added parentheses around TermCriteria
            cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))  # FIX: Added parentheses around TermCriteria

            # Append the object points and image points for both cameras
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            valid_image_pairs += 1
        else:
            print(f"Checkerboard not found in {left_img} and {right_img}")

# If no valid image pairs, print a message
if valid_image_pairs == 0:
    print("No valid image pairs found for calibration.")
else:
    print(f"Found {valid_image_pairs} valid image pairs for calibration.")

# Proceed with calibration only if we have valid image pairs
if valid_image_pairs > 0:
    # Perform camera calibration for both cameras
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5
    # Perform stereo calibration
# Update calibration criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)  # Increase iterations and tolerance
    # ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, leftmatrix, leftDist, rightmatrix, rightDist, gray1.shape[::-1], criteria=criteria,flags = 0)
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1], criteria=criteria,flags = flags
)
    
    # Calculate the average reprojection error for the left and right images
    


    # Print the results to the console
    print("\n--- Calibration Results ---")
    print("Camera 1 Matrix:")
    print(mtx1)
    print("Camera 1 Distortion Coefficients:")
    print(dist1)
    print("Camera 2 Matrix:")
    print(mtx2)
    print("Camera 2 Distortion Coefficients:")
    print(dist2)
    print("Stereo Rotation Matrix:")
    print(R)
    print("Stereo Translation Vector:")
    print(T)
    print("Stereo Essential Matrix:")
    print(E)
    print("Stereo Fundamental Matrix:")
    print(F)

    # Save the calibration results for later use
    np.savez("stereo_calibration.npz", mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, R=R, T=T, E=E, F=F)

    print("\nCalibration data saved to 'stereo_calibration.npz'")

    total_error = 0
    for i in range(valid_image_pairs):
        imgpoints1_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs1[i], tvecs1[i], mtx1, dist1)
        imgpoints2_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs2[i], tvecs2[i], mtx2, dist2)
        
        error1 = cv2.norm(imgpoints1[i], imgpoints1_reprojected, cv2.NORM_L2) / len(imgpoints1[i])
        error2 = cv2.norm(imgpoints2[i], imgpoints2_reprojected, cv2.NORM_L2) / len(imgpoints2[i])
        
        total_error += (error1 + error2)

    average_error = total_error / (2 * valid_image_pairs)
    print(f"Average Reprojection Error: {average_error}")
else:
    print("Stereo calibration failed due to lack of valid image pairs.")


