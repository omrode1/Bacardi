import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

def create_train_val_split(source_dir, output_dir, split_ratio=0.8):
    """
    Splits the dataset into training and validation sets.

    Args:
        source_dir (str): Directory containing the dataset.
        output_dir (str): Directory where the train/val folders will be created.
        split_ratio (float): Ratio of train/val split (default is 0.8 for 80% train, 20% val).
    """

    # Define paths for the train and val folders
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List the class folders inside the source directory
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in classes:
        class_path = Path(source_dir) / class_name
        images = os.listdir(class_path)

        # Shuffle the list of images to ensure randomness
        random.shuffle(images)

        # Split the images into train and val based on the split ratio
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create class folders in train/ and val/
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy train images
        for img in train_images:
            src_img_path = class_path / img
            dst_img_path = train_class_dir / img
            shutil.copy(src_img_path, dst_img_path)

        # Copy val images
        for img in val_images:
            src_img_path = class_path / img
            dst_img_path = val_class_dir / img
            shutil.copy(src_img_path, dst_img_path)

        print(f"Class '{class_name}' - {len(train_images)} train, {len(val_images)} val images")

if __name__ == "__main__":
    # Define the source directory where your dataset is currently located
    source_dataset_dir = "/home/quantic/Om-Projects/clients/Bacardi/data/images/Bacardi_other_cls"

    # Define the output directory where the train/ and val/ folders will be created
    output_dir = "/home/quantic/Om-Projects/clients/Bacardi/data/images/Bacardi_other_cls_split"

    # Set the desired train/val split ratio (e.g., 80% train, 20% val)
    split_ratio = 0.8

    # Call the function to create the train/val split
    create_train_val_split(source_dataset_dir, output_dir, split_ratio)
