# Stereo Vision Toolkit

This project provides a set of Python scripts for stereo vision processing using OpenCV. It covers key components like camera calibration, frame extraction, stereo rectification, and depth estimation.

## 📁 Contents

- `cam_calibration.py`: Calibrates a single camera using chessboard patterns.
- `extract frames.py`: Extracts frames from a stereo video sequence.
- `stereo calibrate.py`: Calibrates a stereo camera setup.
- `Stereo_Rectify.py`: Performs stereo image rectification.
- `stereo.py`: Stereo image capture and basic processing.
- `stereo_estimation.py` / `stereo_estimation_2.py`: Computes disparity and depth maps.
- `stereo-processed-disparity.py`: Further processes and visualizes disparity maps.
- `target.py`: Utility or target generation script.
- `test.py`: For testing components or debugging.

## 🛠 Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
