import os
import requests
from serpapi.google_search import GoogleSearch
from PIL import Image
from io import BytesIO
import time

# Set your SerpAPI key
SERPAPI_KEY = "72ffb09dd3be872aa8d56518d69cd515f1dc165613467fe956f7d7f9da79d97d"

# Folder structure based on your Bacardi types
bacardi_types = [
     "Bacardi Raspberry", 
    # "Bacardi Spiced", "Bacardi Superior"
]

# Directory to save images
dataset_dir = "dataset-collected"

# Create folders for each Bacardi type
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

for bacardi_type in bacardi_types:
    folder_path = os.path.join(dataset_dir, bacardi_type)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to fetch high-resolution images using SerpAPI
def fetch_bottle_images(query, save_folder, num_images=50):
    search_params = {
        "q": query + " bottle",
        "tbm": "isch",
        "ijn": "0",  # page number
        "api_key": SERPAPI_KEY,
        "filter": "0",  # Remove size filter to include high-res images
    }

    search = GoogleSearch(search_params)
    results = search.get_dict()

    images_downloaded = 0

    for image_info in results.get("images_results", []):
        if images_downloaded >= num_images:
            break
        image_url = image_info.get("original")
        
        if image_url:
            try:
                # Fetch the image content
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))

                # Save the image in JPEG or PNG format (YOLOv8-compatible)
                image_name = f"{query.replace(' ', '_')}_{images_downloaded + 1}.jpg"
                image.save(os.path.join(save_folder, image_name), "JPEG")

                images_downloaded += 1
                print(f"Downloaded: {image_name}")

                # Adding a small delay to avoid overloading the API
                time.sleep(1)

            except Exception as e:
                print(f"Error downloading {image_url}: {e}")

# Loop through Bacardi types and fetch images
for bacardi_type in bacardi_types:
    folder_path = os.path.join(dataset_dir, bacardi_type)
    fetch_bottle_images(bacardi_type, folder_path, num_images=100)

print("Image collection completed.")
