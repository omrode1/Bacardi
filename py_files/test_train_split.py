import os
import shutil
import random

# Paths to images and labels
images_dir = '/home/quantic/Om-Projects/clients/Bacardi/data/TrainFlavors_oct21'  # Path to your images directory
labels_dir = '/home/quantic/Om-Projects/clients/Bacardi/data/TrainFlavors_oct21'  # Path to your labels directory

# Output directories for train/val split
output_dir = '/home/quantic/Om-Projects/clients/Bacardi/data/TrainFlavors_oct21_split'  # Output directory
train_images_dir = os.path.join(output_dir, 'images', 'train')
train_labels_dir = os.path.join(output_dir, 'labels', 'train')
val_images_dir = os.path.join(output_dir, 'images', 'val')
val_labels_dir = os.path.join(output_dir, 'labels', 'val')

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Split ratio for train and validation (80% train, 20% validation)
split_ratio = 0.8

# List all images
all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Shuffle the images for random splitting
random.shuffle(all_images)

# Calculate split index
split_index = int(len(all_images) * split_ratio)

# Create train and validation splits
train_images = all_images[:split_index]
val_images = all_images[split_index:]

def copy_files(image_list, images_dest, labels_dest):
    for img_file in image_list:
        label_file = img_file.replace('.jpg', '.txt')

        # Source paths
        img_src = os.path.join(images_dir, img_file)
        label_src = os.path.join(labels_dir, label_file)

        # Destination paths
        img_dest = os.path.join(images_dest, img_file)
        label_dest = os.path.join(labels_dest, label_file)

        # Copy image and label
        shutil.copy(img_src, img_dest)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dest)

# Copy files to train and validation directories
copy_files(train_images, train_images_dir, train_labels_dir)
copy_files(val_images, val_images_dir, val_labels_dir)

print(f"Train set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
