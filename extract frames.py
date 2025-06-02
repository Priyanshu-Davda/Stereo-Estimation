import cv2
import os

def extract_frames(video_path, output_dir):
    # Create directory to store frames if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Frame counter
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every frame as an image
        frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        count += 1
    
    cap.release()
    print(f"Frames extracted and saved to {output_dir}")

# Paths to your stereo video files
# video1_path = 'stereo_video_1_old.avi'  # Replace with the actual path
# video2_path = 'stereo_video_2_old.avi'  # Replace with the actual path

video1_path = 'stereo_video_1.avi'  # Replace with the actual path
video2_path = 'stereo_video_2.avi'  # Replace with the actual path

# video1_path = 'left_output.mp4'  # Replace with the actual path
# video2_path = 'right_output.mp4'  # Replace with the actual path

# Directories to store the extracted frames
output_dir_left = 'left_frames'
output_dir_right = 'right_frames'

# Extract frames for both videos
extract_frames(video1_path, output_dir_left)
extract_frames(video2_path, output_dir_right)
