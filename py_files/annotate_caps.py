import cv2
import os
import sys

def convert_to_yolo_format(image_width, image_height, bbox):
    """Convert bounding box coordinates to YOLO format."""
    x_center = (bbox[0] + bbox[2]) / 2 / image_width
    y_center = (bbox[1] + bbox[3]) / 2 / image_height
    width = (bbox[2] - bbox[0]) / image_width
    height = (bbox[3] - bbox[1]) / image_height
    return x_center, y_center, width, height

# Command line argument for class ID
if len(sys.argv) != 2:
    print("Usage: python script.py <class_id>")
    sys.exit(1)

class_id = int(sys.argv[1])

# Define the folder containing images
input_folder = 'dataset-collected/All-cropped-segregated/Superior'
output_folder = 'dataset-collected/All-cropped-segregated/Superior'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other formats if needed
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            continue  # Skip if the image is not loaded

        # Get image dimensions
        height, width = image.shape[:2]

        # Define bounding box coordinates (top 22% starting from 2% down, cut 1% from both sides)
        start_y = int(height * 0.005)  # 2% from the top
        end_y = int(height * 0.40)     # 24% from the top (2% + 22%)
        start_x = int(width * 0.01)    # 1% from the left
        end_x = int(width * 0.99)       # 1% from the right

        # Draw the bounding box on the original image (optional)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Convert to YOLO format
        bbox = [start_x, start_y, end_x, end_y]  # [x1, y1, x2, y2]
        yolo_bbox = convert_to_yolo_format(width, height, bbox)

        # Save YOLO format annotation
        yolo_annotation_path = os.path.join(output_folder, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        with open(yolo_annotation_path, 'w') as f:
            f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        # Optional: Display the image with bounding box
        cv2.imshow('Image with Bounding Box', image)
        cv2.waitKey(100)  # Display each image for 100 ms

cv2.destroyAllWindows()
