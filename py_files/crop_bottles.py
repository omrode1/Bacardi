import os
import cv2
from ultralytics import YOLO

# Load YOLOv8 model (COCO pre-trained model)
model = YOLO('yolov8m.pt')  # Using YOLOv8 Nano for detection, you can replace this with a different YOLOv8 model if needed

# Function to crop detected objects and save them
def crop_and_save(image, bboxes, output_folder, image_name):
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox[:4])  # Bounding box coordinates
        cropped_image = image[y1:y2, x1:x2]
        
        # Save cropped image
        output_path = os.path.join(output_folder, f"{image_name}_crop_{idx}.jpg")
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved at {output_path}")

# Function to process images from input folder and save cropped objects to output folder
def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the input folder
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)

        # Check if it's an image file (by extension)
        if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"Processing {image_file}...")
            
            # Load image
            image = cv2.imread(image_path)
            
            # Perform detection with YOLOv8
            results = model(image, conf=0.6)
            
            # Filter detections for bottles (class ID for bottle in COCO is 39)
            bboxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if int(box.cls) == 39]  # class ID for bottle
            
            if bboxes:
                crop_and_save(image, bboxes, output_folder, image_file.split('.')[0])
            else:
                print(f"No bottles detected in {image_file}")

# Input and output folders
input_folder = '/home/quantic/Downloads/spiced'    # Replace with your input folder path containing images
output_folder = 'dataset-collected/All-cropped-segregated/Spiced'  # Replace with your output folder path to save cropped images

# Process images
process_images(input_folder, output_folder)
