''' 
file name = generate_modified.py

'''
import os
import random
import shutil
from PIL import Image

# using relative paths 
#os.makedirs("malle_dataset/original_images", exist_ok=True)

# source paths to files 
malle_path = 'malle_dataset/'

random.seed(17) # for deterministic results 

#
original_images = os.listdir(f'{malle_path}original_images/')
#print( original_images)

image_operations = [
    {
        'name': 'cropping',
        'operations': {
            'center_crop': {
                'params': {
                    'width': 256,
                    'height': 256
                }
            },
            'random_crop': {
                'params': {
                    'width': 256,
                    'height': 256
                }
            }
        }
    },
    {
        'name': 'resizing',
        'operations': {
            'downsample': {
                'params': {
                    'target_width': 256,
                    'target_height': 256
                }
            }
        }
    },
    {
        'name': 'rotation',
        'operations': {
            'small_angle': {
                'params': {
                    'angle_range': (-15, 15)
                }
            }
        }
    },
    {
        'name': 'blur',
        'operations': {
            'gaussian': {
                'params': {
                    'radius': 2
                }
            }
        }
    },
    {
        'name': 'brightness_contrast',
        'operations': {
            'adjust': {
                'params': {
                    'brightness_factor': (0.8, 1.2),
                    'contrast_factor': (0.6, 1.4)
                }
            }
        }
    },
    {
        'name': 'color',
        'operations': {
            'shift': {
                'params': {
                    'hue_shift': (-10, 10),
                    'saturation_factor': (0.6, 1.4)
                }
            }
        }
    },
    {
        'name': 'watermark',
        'operations': {
            'overlay': {
                'params': {
                    'text_or_logo': 'SampleLogo',
                    'position': (10, 10),
                    'opacity': 0.3
                }
            }
        }
    },
    {
        'name': 'compression',
        'operations': {
            'jpeg_artifacts': {
                'params': {
                    'quality': (30, 60)
                }
            }
        }
    },
    {
        'name': 'occlusion',
        'operations': {
            'partial_mask': {
                'params': {
                    'mask_size': (50, 50),
                    'mask_position': (100, 100)
                }
            }
        }
    }
]
#  possible values for the number of modified copies for each image
poss_num_mods_per_img = [4,8]

modify_depth = [] # how many different modifications do we want to apply to one image 


for pic in original_images:
    # pick the number of modified copies to genenrate for this image
    num_mods = random.choice(poss_num_mods_per_img)
    
    
    for mod in num_mods:
        # pick the number of different modifications to perform per copy
        mod_depth = random.choice(range(1,9)) 
        
        # choose the modifications to be performed 
        mod_list = random.sample(image_operations, k = mod_depth)

        for mod in mod_list:
            if mod['name'] == 'cropping':

