''' 
file name = extract photos.py
This code would extract 'num_images' from the mini imagenet dataset 
and save them in 'malle_dataset/orginal_images'

'''
import os
import random
import shutil

# using relative paths 
os.makedirs("malle_dataset/original_images", exist_ok=True)

# source paths to files 
imgnet_path = 'imagenet-mini/'
malle_path = 'malle_dataset/'

# list of folder names in the train folder of imagenet-mini
imgnet_folder_names = os.listdir(f'{imgnet_path}train')

# to be deterministic - all calls to random should give the same result everytime 
random.seed(17)

num_images = 200

# the number of different image classes we want to use 
num_classes = 20

#select folder names at random 
selected_folders = random.sample(imgnet_folder_names, num_classes)
#print( os.listdir(imgnet_path + 'train'))

# for each folder, extract name of all files in it, randomly select 
# (num_images/num_classes) files and copy them to 'malle_dataset\orginal_images'
for folder in selected_folders:
    file_list = os.listdir(imgnet_path + 'train/' + folder)
    selected_files = random.sample(file_list, (num_images//num_classes))
    
    for file in selected_files:
        src = f'{imgnet_path}train/{folder}/{file}'
        dst = f'malle_dataset/original_images/{file}'

        shutil.copyfile(src, dst)

        
