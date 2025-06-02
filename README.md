## Stereo Vision Pipeline using OpenCV

This project implements a complete stereo vision pipeline using OpenCV and Python. It includes camera calibration, stereo image rectification, disparity map generation, and interactive visualization of tuning parameters.

## 📁 Files Overview

### Calibration Scripts
- `cam_calibration.py`: Calibrates a single camera using chessboard images.
- `stereo_calibrate.py`: Performs stereo calibration between left and right cameras.
- `Stereo_Rectify.py`: Rectifies stereo image pairs using calibration data.
- `target.py`: Exports detected chessboard corners to an Excel file.

### Stereo Capture and Processing
- `stereo.py`: Captures stereo video from dual cameras and saves as a video file.
- `extract_frames.py`: Extracts left and right frames from saved stereo videos.
- `stereo_estimation.py` / `stereo_estimation_2.py`: Computes disparity and depth maps (early versions).
- `stereo-processed-disparity.py`: Interactive disparity map viewer with real-time parameter tuning.
- `test.py`: Utility script for testing features individually.

### 🔄 Pipeline Overview
---------------------
1.	Capture stereo video using `stereo.py`
2.	Extract left/right frames using `extract_frames.py`
3.	Calibrate cameras using `cam_calibration.py` and `stereo_calibrate.py`
4.	Rectify frames using `Stereo_Rectify.py`
5.	Estimate disparity map using `stereo-processed-disparity.py` or `stereo_estimation.py`

   
### 🖼️ Interactive Disparity Viewer
-------------------------------
Use `stereo-processed-disparity.py` to:
  1. Load rectified grayscale images.
  2. Preprocess images (bilateral filter, CLAHE, NLM, sharpening).
  3. Adjust StereoSGBM parameters in real time.
  4. View raw disparity and filtered outputs. 
  
### To run:

python stereo-processed-disparity.py
Edit the `left_path` and `right_path` variables in the script to use your own images.


## 🛠 Requirements

- Python 3.x
- OpenCV (contrib version)
- NumPy

Install dependencies:

```bash
pip install opencv-contrib-python numpy
```


### 📌 Notes

All disparity calculations require rectified grayscale stereo images.
For calibration, use consistent chessboard patterns with accurate scaling.

### License
-------
MIT License
