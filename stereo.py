import cv2
import numpy as np
import time
# Set up camera indices for your two cameras
camera1_index = 0  # Typically, 0 is the first camera, 1 is the second, etc.
camera2_index = 1  # Set this to the index of your second camera
# Open the video capture for both cameras
cap1 = cv2.VideoCapture(camera1_index)
cap2 = cv2.VideoCapture(camera2_index)
# Check if the cameras opened correctly
if cap1.isOpened():
    print(f"Camera {camera1_index} opened successfully.")
else:
    print(f"Camera {camera1_index} failed to open.")
if cap2.isOpened():
    print(f"Camera {camera2_index} opened successfully.")
else:    print(f"Camera {camera2_index} failed to open.")   
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Couldn't open one or both cameras.")
    exit()
# Get the frames per second (FPS) and resolution from the first camera (assuming both cameras have the same FPS and resolution)
fps = cap1.get(cv2.CAP_PROP_FPS)
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create VideoWriter objects to save the videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('stereo_video_1.avi', fourcc, fps, (frame_width, frame_height))
out2 = cv2.VideoWriter('stereo_video_2.avi', fourcc, fps, (frame_width, frame_height))
print(frame_height, frame_width, fps)
# time.sleep(5)  # Allow cameras to warm up before starting the recording

print("Recording... Press 'q' to stop.")
starttime = time.time()
while True:
    
    # Capture frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("Error: Failed to capture frames.")
        break
    # Write the frames to the respective video files
    out1.write(frame1)
    out2.write(frame2)
    # Optionally, display the frames to ensure synchronization
    # Stack the frames side by side to view both video feeds
    stereo_frame = np.hstack((frame1, frame2))
    cv2.imshow('Stereo Video', stereo_frame)
    elapsedtime = (time.time() - starttime)
    # if elapsedtime >= 50:
    #     print("Stopping video after 10 seconds.")
    #     break
    
    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release resources
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()