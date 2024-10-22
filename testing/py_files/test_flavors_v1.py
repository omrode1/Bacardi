import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 COCO model for detecting bottles
coco_model = YOLO('yolov8n.pt', verbose=False)  # Pre-trained COCO model (can detect bottles)

# Load custom Bacardi classification YOLO model
classification_model = YOLO('testing/models/bacardi_cls_oct21.pt', task='classify')  # Replace with your Bacardi classification model path

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

# Function to classify bottles
def classify_bottle(cropped_bottles):
    classification_results = []
    for i, crop in enumerate(cropped_bottles):
        if crop.size > 0:  # Ensure the cropped image is valid
            try:
                results = classification_model(crop, verbose = False)  # Perform inference

                # Debugging: Check if results are None
                if results is None:
                    print(f"Classification model returned None for crop {i}")
                    classification_results.append("Unknown")
                    continue

                # Check if results contain any boxes
                if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # Get the class name with the highest confidence
                    class_name = results[0].names[int(results[0].boxes.cls[0])]
                    classification_results.append(class_name)
                else:
                    print(f"No valid boxes detected for crop {i}")
                    classification_results.append("Unknown")

            except Exception as e:
                print(f"Error during classification for crop {i}: {str(e)}")
                classification_results.append("Error")
        else:
            classification_results.append("Invalid Image")

    return classification_results

def classify_bottle(cropped_bottles):
    classification_results = []

    for i, crop in enumerate(cropped_bottles):
        if crop.size > 0:  # Ensure the cropped image is valid
            try:
                # Resize the cropped image to match the model's input size
                resized_crop = cv2.resize(crop, (128, 128))
                
                # Convert the image to RGB if it's not already
                if len(resized_crop.shape) == 2:
                    resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_GRAY2RGB)
                elif resized_crop.shape[2] == 4:
                    resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_RGBA2RGB)

                # Convert the resized image to the format expected by YOLO
                # The model takes a list of images, so wrap it in a list
                results = classification_model(np.array([resized_crop]))

                # Extract the predicted class (assuming a single class prediction)
                predicted_class = results[0].names[results[0].probs.top1]
                
                classification_results.append(predicted_class)

            except Exception as e:
                print(f"Error during classification for crop {i}: {str(e)}")
                classification_results.append("Error")
        else:
            classification_results.append("Invalid Image")

    return classification_results


# Function to annotate image with flavor labels for Bacardi bottles only
def annotate_image(image, boxes, labels):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Only annotate if the label corresponds to a Bacardi bottle
        if i < len(labels) and labels[i].lower() == "bacardi":
            # Draw the bounding box for Bacardi bottles
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the flavor label if it exists
            label = labels[i] if len(labels) > i else "Unknown"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
    return image






# Function to process both images and videos
def process_files(input_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Check if it's an image
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing image: {filename}")
            #process_image(file_path)
        
        # Check if it's a video
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing video: {filename}")
            process_video(file_path)

# Function to process an image
# def process_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Run bottle detection using YOLOv8 COCO model
#     coco_results = coco_model(image)

#     # Extract bounding boxes from the detection results
#     boxes = coco_results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes: [x1, y1, x2, y2, confidence, class]
    
#     # Crop the detected bottles
#     cropped_bottles, cropped_coords = crop_objects(image, boxes)

#     # Classify bottles as Bacardi or others
#     classification_results = classify_bottle(cropped_bottles)

#     # Only pass Bacardi bottles to the flavor detection model
#     bacardi_bottles = [crop for i, crop in enumerate(cropped_bottles) if classification_results[i] == "bacardi"]
#     if len(bacardi_bottles) > 0:
#         flavor_predictions = detect_flavor(bacardi_bottles)
#     else:
#         flavor_predictions = []

#     # Annotate the image with Bacardi flavors (if detected)
#     final_image = annotate_image(image, cropped_coords, classification_results if len(flavor_predictions) == 0 else flavor_predictions)

#     # Display the result using OpenCV
#     cv2.imshow("Detected Bacardi Flavors - Image", final_image)
#     cv2.waitKey(0)  # Press any key to close the image
#     cv2.destroyAllWindows()




# Function to detect flavors in Bacardi bottles
def detect_flavor(bacardi_bottles):
    flavor_results = []
    for i, bottle in enumerate(bacardi_bottles):
        if bottle.size > 0:  # Ensure the cropped image is valid
            try:
                # Resize the cropped image to match the model's input size
                resized_bottle = cv2.resize(bottle, (224, 224))  # Adjust size according to your model's input size
                
                # Convert the image to RGB if it's not already
                if len(resized_bottle.shape) == 2:
                    resized_bottle = cv2.cvtColor(resized_bottle, cv2.COLOR_GRAY2RGB)
                elif resized_bottle.shape[2] == 4:
                    resized_bottle = cv2.cvtColor(resized_bottle, cv2.COLOR_RGBA2RGB)
                
                # Perform inference
                results = flavor_model(resized_bottle)
                
                # Extract the predicted class
                predicted_class = results[0].probs.top1
                class_name = results[0].names[predicted_class]
                
                flavor_results.append(class_name)

            except Exception as e:
                print(f"Error during flavor detection for bottle {i}: {str(e)}")
                flavor_results.append("Error")
        else:
            flavor_results.append("Invalid Image")

    return flavor_results

# Function to process a video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break  # Break when the video ends

        # Run bottle detection using YOLOv8 COCO model
        coco_results = coco_model(frame)

        # Extract bounding boxes from the detection results
        boxes = coco_results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes: [x1, y1, x2, y2, confidence, class]
        print(f"Detected boxes: {boxes}")

        # Crop the detected bottles
        cropped_bottles, cropped_coords = crop_objects(frame, boxes)
        print(f"Cropped bottles: {len(cropped_bottles)}")

        # Classify bottles as Bacardi or others
        classification_results = classify_bottle(cropped_bottles)
        print(f"Classification results: {classification_results}")

        # Only pass Bacardi bottles to the flavor detection model
        bacardi_bottles = [crop for i, crop in enumerate(cropped_bottles) if classification_results[i].lower() == "bacardi"]
        bacardi_coords = [cropped_coords[i] for i in range(len(cropped_bottles)) if classification_results[i].lower() == "bacardi"]

        if len(bacardi_bottles) > 0:
            flavor_predictions = detect_flavor(bacardi_bottles)
            print(f"Flavor predictions: {flavor_predictions}")
        else:
            flavor_predictions = []

        # Annotate the frame with Bacardi flavors (if detected)
        final_frame = annotate_image(frame, bacardi_coords, classification_results)

        # Display the result using OpenCV
        cv2.imshow("Detected Bacardi Flavors - Video", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit the video

    cap.release()
    cv2.destroyAllWindows()


# Process all files in the input folder
process_files(input_folder)
