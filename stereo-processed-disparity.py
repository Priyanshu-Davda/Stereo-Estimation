import cv2
import numpy as np

# === Load rectified stereo pair (grayscale) ===
left_path = 'rectified_left_image.jpg'
right_path = 'rectified_right_image.jpg'

imgL_orig = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
imgR_orig = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
if imgL_orig is None or imgR_orig is None:
    raise IOError("Could not load rectified images. Check file paths.")

# === Window & Trackbar Setup ===
cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)

# Preprocessing parameters
cv2.createTrackbar("Bilateral d (×2+1)", "Settings", 2, 10, lambda x: None)        # d = x*2+1
cv2.createTrackbar("Bilateral σColor", "Settings", 75, 200, lambda x: None)
cv2.createTrackbar("Bilateral σSpace", "Settings", 75, 200, lambda x: None)
cv2.createTrackbar("NLM h", "Settings", 10, 30, lambda x: None)
cv2.createTrackbar("NLM tmplSz", "Settings", 7, 20, lambda x: None)               # ensure odd
cv2.createTrackbar("NLM searchSz", "Settings", 21, 50, lambda x: None)           # ensure odd
cv2.createTrackbar("CLAHE clip (×0.1)", "Settings", 20, 40, lambda x: None)       # clip = x/10
cv2.createTrackbar("CLAHE grid Sz", "Settings", 8, 16, lambda x: None)            # tileGridSize
cv2.createTrackbar("Unsharp factor (×0.01)", "Settings", 50, 100, lambda x: None) # factor = x/100

# StereoSGBM parameters
cv2.createTrackbar("minDisparity", "Settings", 0, 50, lambda x: None)
cv2.createTrackbar("numDisp (×16)", "Settings", 6, 15, lambda x: None)            # numDisp = x*16
cv2.createTrackbar("blockSize (odd)", "Settings", 9, 51, lambda x: None)          # ensure odd≥1
cv2.createTrackbar("uniquenessRatio", "Settings", 15, 50, lambda x: None)
cv2.createTrackbar("speckleWindow", "Settings", 100, 200, lambda x: None)
cv2.createTrackbar("speckleRange", "Settings", 32, 50, lambda x: None)
cv2.createTrackbar("preFilterCap", "Settings", 63, 100, lambda x: None)
cv2.createTrackbar("P1 factor", "Settings", 8, 50, lambda x: None)                # P1 = val * bs²
cv2.createTrackbar("P2 factor", "Settings", 32, 100, lambda x: None)              # P2 = val * bs²

# Create display windows
cv2.namedWindow("Left Processed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Processed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

def preprocess(img):
    """
    Returns the final 'sharp' image after:
    1. Bilateral filtering
    2. Fast NLM denoising
    3. CLAHE
    4. Unsharp mask
    """
    # Read sliders
    d      = cv2.getTrackbarPos("Bilateral d (×2+1)", "Settings") * 2 + 1
    sigmaC = cv2.getTrackbarPos("Bilateral σColor", "Settings")
    sigmaS = cv2.getTrackbarPos("Bilateral σSpace", "Settings")
    h_nlm  = cv2.getTrackbarPos("NLM h", "Settings")
    tmpl   = cv2.getTrackbarPos("NLM tmplSz", "Settings")
    search  = cv2.getTrackbarPos("NLM searchSz", "Settings")
    clip   = max(cv2.getTrackbarPos("CLAHE clip (×0.1)", "Settings") / 10.0, 0.1)
    grid   = max(cv2.getTrackbarPos("CLAHE grid Sz", "Settings"), 1)
    factor = cv2.getTrackbarPos("Unsharp factor (×0.01)", "Settings") / 100.0

    # 1. Bilateral Filter
    bilat = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)

    # 2. Fast Non-Local Means Denoising
    tmpl = tmpl if tmpl % 2 == 1 and tmpl >= 1 else tmpl + 1
    search = search if search % 2 == 1 and search >= 1 else search + 1
    denoised = cv2.fastNlMeansDenoising(bilat, None, h=h_nlm, templateWindowSize=tmpl, searchWindowSize=search)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    enhanced = clahe.apply(denoised)

    # 4. Unsharp Mask
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    sharp = cv2.addWeighted(enhanced, 1.0 + factor, blur, -factor, 0)

    return sharp

def compute_disparity(left_img, right_img):
    """
    Computes disparity map using StereoSGBM with parameters from sliders.
    Returns raw_disp (fixed-point int16) and disp (float32 /16.0).
    """
    # Read sliders
    min_disp = cv2.getTrackbarPos("minDisparity", "Settings")
    num_disp = cv2.getTrackbarPos("numDisp (×16)", "Settings") * 16
    bs       = cv2.getTrackbarPos("blockSize (odd)", "Settings")
    bs = bs if bs % 2 == 1 and bs >= 1 else bs + 1
    uniR     = cv2.getTrackbarPos("uniquenessRatio", "Settings")
    spW      = cv2.getTrackbarPos("speckleWindow", "Settings")
    spR      = cv2.getTrackbarPos("speckleRange", "Settings")
    pfc      = cv2.getTrackbarPos("preFilterCap", "Settings")
    p1f      = cv2.getTrackbarPos("P1 factor", "Settings")
    p2f      = cv2.getTrackbarPos("P2 factor", "Settings")

    # Calculate P1, P2
    P1 = p1f * bs * bs
    P2 = p2f * bs * bs

    # Create StereoSGBM matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = max(16, num_disp),
        blockSize = bs,
        P1 = P1,
        P2 = P2,
        disp12MaxDiff = 1,
        uniquenessRatio = uniR,
        speckleWindowSize = spW,
        speckleRange = spR,
        preFilterCap = pfc,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    raw_disp = stereo.compute(left_img, right_img)
    disp = raw_disp.astype(np.float32) / 16.0
    return raw_disp, disp

while True:
    # 1. Preprocess both images
    left_proc  = preprocess(imgL_orig)
    right_proc = preprocess(imgR_orig)

    # Display processed left and right
    cv2.imshow("Left Processed", left_proc)
    cv2.imshow("Right Processed", right_proc)

    # 2. Compute disparity on the 'sharp' outputs
    raw_disp, disp = compute_disparity(left_proc, right_proc)

    # Show raw disparity (normalized for display)
    raw_vis = cv2.normalize(raw_disp, None, 0, 255, cv2.NORM_MINMAX)
    raw_vis = np.uint8(raw_vis)
    cv2.imshow("Disparity", raw_vis)

    # Listen for ESC key to exit
    key = cv2.waitKey(50) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
