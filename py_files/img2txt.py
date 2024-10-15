from ultralytics import YOLO
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
import os

# Load the Moondream v2 model and tokenizer from the local directory
model_path = os.path.abspath("/home/quantic/Om-Projects/clients/Bacardi/models/img2txt")
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the YOLOv8 model pretrained on the COCO dataset
model_yolo = YOLO('yolov8n.pt')

# Load your image
image_path = '/home/quantic/Om-Projects/clients/Bacardi/cropped_bottle_1.jpg'
image = cv2.imread(image_path)

# Perform inference on the image
results = model_yolo(image)

# Extract detected objects (use `boxes` for the updated YOLOv8 result format)
detected_objects = results[0].boxes

# Class 39 corresponds to 'bottle' in the COCO dataset
bottle_class_id = 39

# Filter the results to only include bottles
bottles = [obj for obj in detected_objects if int(obj.cls[0]) == bottle_class_id]

# Sort bottles by the x-coordinate of the bounding box (left to right)
bottles_sorted = sorted(bottles, key=lambda b: b.xyxy[0][0].item())

# Initialize a list to store OCR results
ocr_results = []

# Loop through each detected bottle in left-to-right order
for bottle_id, bottle in enumerate(bottles_sorted, start=1):
    x1, y1, x2, y2 = map(int, bottle.xyxy[0])  # Extract the coordinates
    
    # Crop the image to the bounding box of the detected bottle
    cropped_bottle = image[y1:y2, x1:x2]
    
    # Convert cropped bottle from OpenCV format (BGR) to PIL format (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(cropped_bottle, cv2.COLOR_BGR2RGB))
    
    # Encode the cropped bottle image using Moondream's encode_image function
    enc_image = model.encode_image(pil_image)
    
    # Ask the model to describe the image (OCR or details)
    ocr_output = model.answer_question(enc_image, "Extract OCR data", tokenizer)
    
    # Store the bottle ID and OCR output in the list
    ocr_results.append((bottle_id, ocr_output.strip()))
    
    # Draw rectangle around the detected bottle and show ID
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display bottle ID on the original image
    label = f"Bottle ID: {bottle_id}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Optionally save each cropped bottle with its respective ID
    cv2.imwrite(f'cropped_bottle_{bottle_id}.jpg', cropped_bottle)

print("OCR Results for Each Bottle (from left to right):")
for bottle_id, ocr_output in ocr_results:
    print(f"Bottle ID {bottle_id}: {ocr_output}")


# Show the image with detections and IDs in an OpenCV window
cv2.imshow('Detected Bottles with IDs', image)

# Wait until any key is pressed, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image if needed
cv2.imwrite('output_bottle_detection_with_ids.jpg', image)

# Print the OCR results in the terminal
# print("OCR Results for Each Bottle (from left to right):")
# for bottle_id, ocr_output in ocr_results:
#     print(f"Bottle ID {bottle_id}: {ocr_output}")
