import os
import shutil

# Define the target classes
classes = ['Limon', 'Black', 'Apple', 'Gold', 'Superior', 'Spiced']

# Input folder containing cropped images
cropped_folder = '/home/quantic/Om-Projects/clients/Bacardi/dataset-collected/All-cropped'  # Replace with your cropped images folder path

# Base folder where you want to separate images into class folders
output_base_folder = '/home/quantic/Om-Projects/clients/Bacardi/dataset-collected/All-cropped-segregated'  # Replace with the desired output base folder

# Function to create class folders if they don't exist
def create_class_folders(base_folder, class_names):
    for class_name in class_names:
        class_folder = os.path.join(base_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

# Function to organize cropped images based on class names
def organize_images_by_class(cropped_folder, output_base_folder, class_names):
    # Create folders for each class
    create_class_folders(output_base_folder, class_names)
    
    # Loop over all images in the cropped folder
    for image_file in os.listdir(cropped_folder):
        image_path = os.path.join(cropped_folder, image_file)

        # Check if the image filename contains any class name
        class_found = False
        for class_name in class_names:
            if class_name.lower() in image_file.lower():
                class_folder = os.path.join(output_base_folder, class_name)
                shutil.copy(image_path, class_folder)
                print(f"Moved {image_file} to {class_folder}")
                class_found = True
                break
        
        # If no class name is found, omit the image
        if not class_found:
            print(f"Omitting {image_file} - No class match found.")

# Organize images based on class
organize_images_by_class(cropped_folder, output_base_folder, classes)
