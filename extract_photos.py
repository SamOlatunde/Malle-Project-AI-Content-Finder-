"""
extract_photos.py
Extracts 'num_images' images from mini-imagenet and saves them in malle_dataset/original_images.
"""

import os
import random
import shutil

# Make output folder
os.makedirs('malle_dataset/original_images', exist_ok=True)

imgnet_path = 'imagenet-mini/'
output_path = 'malle_dataset/original_images/'

# List of ALL folders in val (1000 classes)
all_folders = os.listdir(f'{imgnet_path}val')

random.seed(17)

num_images = 200
num_classes = 20

# Step 1: Select 20 folders
selected_folders = random.sample(all_folders, num_classes)

collected = []

# ---------------- FIRST PASS: Collect all images from selected folders ----------------
for folder in selected_folders:
    folder_path = f'{imgnet_path}val/{folder}'
    file_list = os.listdir(folder_path)

    for file in file_list:
        src = f'{folder_path}/{file}'
        file, ext = ((file.split('_'))[-1]).rsplit('.',1)
        correct_file_name = f'{folder}_{file}.{ext}'
        dst = f'{output_path}{correct_file_name}'   # prevent collisions
        shutil.copyfile(src, dst)
        collected.append(src)

print(f"Collected {len(collected)} images from selected 20 folders.")

# ---------------- SECOND PASS: Fill remaining images from ALL folders ----------------
current_count = len(os.listdir(output_path))
remaining_needed = num_images - current_count

print(f"Need {remaining_needed} more images to reach {num_images}.")

if remaining_needed > 0:
    # Build list of ALL image paths
    all_image_paths = []
    for folder in all_folders:
        folder_path = f'{imgnet_path}val/{folder}'
        for file in os.listdir(folder_path):
            all_image_paths.append((folder, f'{folder_path}/{file}'))

    # Exclude duplicates (already copied)
    used_set = set(collected)
    remaining_pool = [(folder, path) for (folder, path) in all_image_paths if path not in used_set]

    # Now we DO have enough images
    extra_images = random.sample(remaining_pool, remaining_needed)

    for folder, src in extra_images:
        file = os.path.basename(src)
        file, ext = ((file.split('_'))[-1]).rsplit('.',1)
        correct_file_name = f'{folder}_{file}.{ext}'
        dst = f'{output_path}{correct_file_name}'   # prevent collisions
        shutil.copyfile(src, dst)

# ---------------- DONE ----------------
print(f"Final image count: {len(os.listdir(output_path))}")
