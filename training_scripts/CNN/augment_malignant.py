import os
import random
import shutil
from PIL import Image, ImageEnhance
import numpy as np

# --- Paths ---
source_dir = 'data/HAM10000/skin_cancer_data'
target_dir = 'data/HAM10000/augmented_data'

# Clean old folder if it exists
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

os.makedirs(os.path.join(target_dir, 'benign'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'malignant'), exist_ok=True)

# --- Augmentation functions ---
def random_rotation(img):
    """Rotate image by a random angle between -30 and 30 degrees"""
    angle = random.uniform(-30, 30)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

def random_flip(img):
    """Randomly flip image horizontally and/or vertically"""
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def random_zoom(img):
    """Randomly zoom into the image by 10-30%"""
    w, h = img.size
    zoom_factor = random.uniform(1.1, 1.3)
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)

def random_brightness(img):
    """Randomly adjust brightness by ±20%"""
    factor = random.uniform(0.8, 1.2)
    return ImageEnhance.Brightness(img).enhance(factor)

def augment_image(img):
    """Apply a random combination of augmentations"""
    transforms = [random_rotation, random_flip, random_zoom, random_brightness]
    # apply 2-3 random transforms
    num_transforms = random.randint(2, 3)
    selected = random.sample(transforms, num_transforms)
    for transform in selected:
        img = transform(img)
    return img


# --- Step 1: Copy ALL benign images (no changes) ---
benign_src = os.path.join(source_dir, 'benign')
benign_files = os.listdir(benign_src)
num_benign = len(benign_files)

print(f"Copying {num_benign} benign images (unchanged)...")
for f in benign_files:
    shutil.copy(os.path.join(benign_src, f), os.path.join(target_dir, 'benign', f))

# --- Step 2: Copy ALL malignant originals ---
malignant_src = os.path.join(source_dir, 'malignant')
malignant_files = os.listdir(malignant_src)
num_malignant = len(malignant_files)

print(f"Copying {num_malignant} malignant originals...")
for f in malignant_files:
    shutil.copy(os.path.join(malignant_src, f), os.path.join(target_dir, 'malignant', f))

# --- Step 3: Augment malignant images until they match benign count ---
num_augmented_needed = num_benign - num_malignant
print(f"Need {num_augmented_needed} augmented malignant images to match {num_benign} benign...")

random.seed(42)
augmented_count = 0

while augmented_count < num_augmented_needed:
    # pick a random malignant image to augment
    src_file = random.choice(malignant_files)
    src_path = os.path.join(malignant_src, src_file)

    try:
        img = Image.open(src_path).convert('RGB')
        aug_img = augment_image(img)

        # save with unique name
        name, ext = os.path.splitext(src_file)
        aug_filename = f"{name}_aug{augmented_count}{ext}"
        aug_img.save(os.path.join(target_dir, 'malignant', aug_filename), quality=95)

        augmented_count += 1

        if augmented_count % 1000 == 0:
            print(f"  ...generated {augmented_count}/{num_augmented_needed} augmented images")

    except Exception as e:
        print(f"  Error augmenting {src_file}: {e}")
        continue

# --- Summary ---
final_benign = len(os.listdir(os.path.join(target_dir, 'benign')))
final_malignant = len(os.listdir(os.path.join(target_dir, 'malignant')))
print(f"\nDone! Final dataset:")
print(f"  Benign:    {final_benign}")
print(f"  Malignant: {final_malignant} ({num_malignant} originals + {augmented_count} augmented)")
print(f"  Total:     {final_benign + final_malignant}")
