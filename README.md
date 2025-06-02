# Stereo Vision Toolkit

This project provides a set of Python scripts for stereo vision processing using OpenCV. It covers key components like camera calibration, frame extraction, stereo rectification, and depth estimation.

## 📁 Contents

- `cam_calibration.py`: Calibrates a single camera using chessboard patterns.
- `extract frames.py`: Extracts frames from a stereo video sequence captured with stereo.py.
- `stereo calibrate.py`: Calibrates a stereo camera setup.
- `Stereo_Rectify.py`: Performs stereo image rectification to be used for exraction.
- `stereo.py`: Stereo video capture and basic processing.
- `stereo_estimation.py` / `stereo_estimation_2.py`: Computes disparity and depth maps (still working on these).
- `stereo-processed-disparity.py`: Further processes and visualizes disparity maps it has sliders for all the paraeters which are used for calculation so that we can understand what parameters is affectiong which aspect of disparity.
- `target.py`: excel generation script for chessboard corners.
- `test.py`: For testing components or debugging jusr for some random codes.


Sample Flow :
Stereo video → Frame Extraction → Stereo Calibration → Rectification → Disparity → Depth Map

Install dependencies:
## 🛠 Requirements

- Python 3.10
- OpenCV (contrib version)
- NumPy

Install dependencies:

```bash
pip install opencv-contrib-python numpy







