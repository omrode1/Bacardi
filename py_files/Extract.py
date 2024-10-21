import cv2
import os

def extract_frames_from_videos(input_folder, output_folder, frame_skip=5):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv', 'webm')):  # Add more video formats if needed
            video_path = os.path.join(input_folder, file_name)
            video_name = os.path.splitext(file_name)[0]


            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {file_name}")
                continue

            frame_count = 0
            saved_frame_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break  # Break when the video ends

                # Check if the current frame should be saved
                if frame_count % frame_skip == 0:
                    output_image_path = os.path.join(output_folder, f"{video_name}_frame_{saved_frame_count}.jpg")
                    cv2.imwrite(output_image_path, frame)
                    print(f"Saved frame {saved_frame_count} from video {file_name}")
                    saved_frame_count += 1

                frame_count += 1

            # Release the video capture object
            cap.release()

    print("Frame extraction completed.")

if __name__ == "__main__":
    # Define the input folder containing videos and the output folder for extracted frames
    input_folder = "/home/quantic/Om-Projects/clients/Bacardi/data/videos"
    output_folder = "/home/quantic/Om-Projects/clients/Bacardi/data/images/uncropped"

    # Define the number of frames to skip between each extraction
    frame_skip = 5  # Extract every 5th frame (you can modify this number)

    extract_frames_from_videos(input_folder, output_folder, frame_skip)
