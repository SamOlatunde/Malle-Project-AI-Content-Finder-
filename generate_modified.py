# generate_modified.py

import os
import random
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms.v2 as transforms

malle_path = "malle_dataset/"

random.seed(17)

original_images = os.listdir(f"{malle_path}original_images/")

img_ops = {
    "cropping": {
        "center_crop": {"params": {"size": [ 256, 384]}},
        "random_crop": {"params": {"size": [ 256, 384]}},
    },
    "resizing": {"params": {"min_size": 256, "max_size": 384}},
    "rotation": {"params": {"degrees": (-15, 15)}},
    "blur": {"params": {"kernel_size": [3, 5,7]}},
    "brightness_contrast": {"params": {"brightness_factor": (0.8, 1.2), "contrast_factor": (0.6, 1.4)}},
    "color": {"params": {"hue_shift": (-0.5, 0.5), "saturation_factor": (0.6, 1.4)}},
    "watermark": {"params": {"text": "Malle Project", "position": (30, 30), "font_size": 80, "opacity": 120}},
    "compression": {"params": {"quality": (30, 80)}},
    "occlusion": {"params": {"mask_size": (80, 80), "mask_position": (90, 90)}},
}

poss_num_mods_per_img = [4, 8]

# --- canonical conversions (IMPORTANT) ---
# PIL -> float tensor in [0,1]
PIL_TO_TENSOR = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)   # <-- scale=True fixes the white/grey problem
])

# tensor in [0,1] -> PIL (works reliably)
TENSOR_TO_PIL = transforms.ToPILImage()

for pic in original_images:
    num_mods = random.choice(poss_num_mods_per_img)

    pic_img = Image.open(f"{malle_path}original_images/{pic}").convert("RGB")

    for mod_num in range(num_mods):
        # start from the original for each copy (and don't accidentally carry over previous mods)
        modified_pic = pic_img.copy()

        # convert once to a consistent tensor domain: float32 in [0,1]
        modified_pic = PIL_TO_TENSOR(modified_pic)

        mods_str = ""
        mod_depth = random.choice(range(1, 8))
        mod_list = random.sample(list(img_ops.keys()), k=mod_depth)

        for mod in mod_list:
            if mod == "cropping":

                crop_flag = random.randint(0, 1)

                if crop_flag == 0:
                    crop_size = random.choice(img_ops["cropping"]["center_crop"]["params"]["size"])
                    modified_pic = transforms.CenterCrop(crop_size)(modified_pic)
                    mods_str += f"_centCrop{crop_size}"
                else:
                    crop_size = random.choice(img_ops["cropping"]["random_crop"]["params"]["size"])
                    modified_pic = transforms.RandomCrop(crop_size, pad_if_needed=True, padding_mode="constant")(modified_pic)
                    mods_str += f"_randCrop{crop_size}"

            elif mod == "resizing":
                modified_pic = transforms.RandomResize(**img_ops["resizing"]["params"])(modified_pic)
                mods_str += "_resize"

            elif mod == "rotation":
                modified_pic = transforms.RandomRotation(img_ops["rotation"]["params"]["degrees"])(modified_pic)
                mods_str += "_rotate"

            elif mod == "blur":
                _, h, w = modified_pic.shape
                choosen_dim = min(h, w)
                if choosen_dim <= 128:
                    kernel_size = img_ops["blur"]["params"]["kernel_size"][0] 
                elif choosen_dim <256:
                    kernel_size = img_ops["blur"]["params"]["kernel_size"][1]
                else:
                    kernel_size = img_ops["blur"]["params"]["kernel_size"][2]
                    
                modified_pic = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size))(modified_pic)
                mods_str += "_blur"

            elif mod == "brightness_contrast":
                bc = transforms.ColorJitter(
                    brightness=img_ops["brightness_contrast"]["params"]["brightness_factor"],
                    contrast=img_ops["brightness_contrast"]["params"]["contrast_factor"],
                )
                modified_pic = bc(modified_pic)
                mods_str += "_brightness_contrast"

            elif mod == "color":
                cj = transforms.ColorJitter(
                    saturation=img_ops["color"]["params"]["saturation_factor"],
                    hue=img_ops["color"]["params"]["hue_shift"],
                )
                modified_pic = cj(modified_pic)
                mods_str += "_color"
                
            elif mod == "watermark":
                # Tensor -> PIL (still safe because tensor is [0,1])
                pil_img = TENSOR_TO_PIL(modified_pic).convert("RGBA")

                watermark_base = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(watermark_base)

                # load_default() ignores size; use a truetype if available; fallback is fine
                try:
                    font = ImageFont.truetype("arial.ttf", img_ops["watermark"]["params"]["font_size"])
                except OSError:
                    font = ImageFont.load_default()

                text = img_ops["watermark"]["params"]["text"]
                position = img_ops["watermark"]["params"]["position"]
                opacity = img_ops["watermark"]["params"]["opacity"]

                draw.text(position, text, fill=(255,255,255,opacity), font=font, stroke_width=3, stroke_fill=(0,0,0))

                watermarked = Image.alpha_composite(pil_img, watermark_base).convert("RGB")

                # Back to consistent tensor domain [0,1]
                modified_pic = PIL_TO_TENSOR(watermarked)
                mods_str += "_watermark"

            elif mod == "compression":
                pil_img = TENSOR_TO_PIL(modified_pic).convert("RGB")

                q_min, q_max = img_ops["compression"]["params"]["quality"]
                q = random.randint(q_min, q_max)

                # JPEG transform expects a quality int (not a tuple)
                modified_pic = PIL_TO_TENSOR(transforms.JPEG(quality=q)(pil_img))
                mods_str += "_compression"

            else:  # occlusion
                x_max, y_max = img_ops["occlusion"]["params"]["mask_position"]
                x = random.randint(0, x_max)
                y = random.randint(0, y_max)
                h, w = img_ops["occlusion"]["params"]["mask_size"]

                # set pixels to 0 (black) in [0,1] space
                modified_pic[:, y:y + h, x:x + w] = 0.0
                mods_str += "_occlusion"

        # Final save: tensor [0,1] -> PIL and write to disk
        final_pil = TENSOR_TO_PIL(modified_pic)

        name, ext = pic.rsplit(".", 1)
        out_path = f"{malle_path}modified_images/{name}{mods_str}.{ext}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final_pil.save(out_path)
