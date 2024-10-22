import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 COCO model for detecting bottles
coco_model = YOLO('yolov8n.pt')  # Pre-trained COCO model (can detect bottles)

# Load custom flavor detection YOLO model
flavor_model = YOLO('testing/models/Flavor_oct21.pt')  # Your custom flavor detection model

# Folder containing videos and images
input_folder = 'testing/videos'  # Modify this with your folder path

# Function to crop the detected objects from the image
def crop_objects(image, boxes):
    cropped_images = []
    cropped_coords = []
    for i, box in enumerate(boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box[:4])  # Assuming [x1, y1, x2, y2]
        cropped_img = image[y1:y2, x1:x2]
        cropped_images.append(cropped_img)
        cropped_coords.append((x1, y1, x2, y2))  # Store the coordinates for later annotation
    return cropped_images, cropped_coords

# Function to run flavor detection on cropped bottle images
def detect_flavor(cropped_images):
    flavor_results = []
    for crop in cropped_images:
        if crop.size > 0:  # Check if the cropped image is valid
            results = flavor_model(crop)

            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the class name with the highest confidence
                class_name = results[0].names[int(results[0].boxes.cls[0])]
                flavor_results.append(class_name)
            else:
                # No detections, append "Unknown"
                flavor_results.append("Unknown")
        else:
            # If the crop is empty, append "Unknown"
            flavor_results.append("Unknown")
    
    return flavor_results


# Function to annotate image with flavor labels
def annotate_image(image, boxes, flavor_labels):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        label = flavor_labels[i]
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display the flavor label
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

# Function to process both images and videos
def process_files(input_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Check if it's an image
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing image: {filename}")
            process_image(file_path)
        
        # Check if it's a video
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing video: {filename}")
            process_video(file_path)

# Function to process an image
def process_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Run bottle detection using YOLOv8 COCO model
    coco_results = coco_model(image)

    # Extract bounding boxes from the detection results
    boxes = coco_results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes: [x1, y1, x2, y2, confidence, class]
    
    # Crop the detected bottles
    cropped_bottles, cropped_coords = crop_objects(image, boxes)
    
    # Detect the flavors in the cropped bottle images
    flavor_predictions = detect_flavor(cropped_bottles)

    # Annotate the original image with the flavor predictions
    final_image = annotate_image(image, boxes, flavor_predictions)

    # Display the result using OpenCV
    cv2.imshow("Detected Flavors - Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process a video
def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break when the video ends

        # Run bottle detection using YOLOv8 COCO model
        coco_results = coco_model(frame)

        # Extract bounding boxes from the detection results
        boxes = coco_results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes: [x1, y1, x2, y2, confidence, class]
        
        # Crop the detected bottles
        cropped_bottles, cropped_coords = crop_objects(frame, boxes)

        # Detect the flavors in the cropped bottle images
        flavor_predictions = detect_flavor(cropped_bottles)

        # Annotate the current video frame with the flavor predictions
        final_frame = annotate_image(frame, boxes, flavor_predictions)

        # Display the result using OpenCV
        cv2.imshow("Detected Flavors - Video", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit the video

    cap.release()
    cv2.destroyAllWindows()

# Run the process
process_files(input_folder)
