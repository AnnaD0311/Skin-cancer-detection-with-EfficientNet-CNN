import os
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse
from preprocessing import preprocess_pipeline


# Configuración de Rutas
BASE_DIR = "."
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")
ORIGINAL_IMAGES_DIRS = [
    os.path.join(BASE_DIR, "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "HAM10000_images_part_2")
]
BALANCED_DIR = os.path.join(BASE_DIR, "balancing", "dataset_balanced")

TARGET_COUNT = 1000

def remove_black_borders(img_pil):
    """
    Automatically detects and crops black bars or edges on the sides/top.    
    """
    img_np = np.array(img_pil)
    
    # Mask pixels that are not completely black (tolerance threshold = 10)
    non_black_mask = np.any(img_np > 10, axis=2)
    
    # Encontrar las coordenadas extremas con información de imagen
    rows = np.any(non_black_mask, axis=1)
    cols = np.any(non_black_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img_pil # Finding extreme coordinates using image information
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Finding extreme coordinates using image information
    cropped_np = img_np[rmin:rmax+1, cmin:cmax+1]
    return Image.fromarray(cropped_np)

def apply_augmentation(img):
    """
    Apply orthogonal rotations (90°, 180°, 270°) and strictly remove any black 
    bars resulting from a rectangular aspect ratio.
    """
    # 1. Choose an orthogonal angle
    angle = random.choice([0, 90, 180, 270])
    
    # 2. Rotate the image (expand=True adjusts the canvas to the new height/width)
    if angle != 0:
        img = TF.rotate(img, angle=angle, expand=True)
    
    # 3. Actively remove black borders, if present
    img = remove_black_borders(img)
    
    # 4. Color inversions and adjustments without distortion
    transforms_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        # Uniform final resizing (e.g., 224x224 for the neural network)
        transforms.Resize((224, 224), interpolation=TF.InterpolationMode.BILINEAR)
    ])
    
    return transforms_pipeline(img)

def find_image_path(image_id):
    for folder in ORIGINAL_IMAGES_DIRS:
        img_path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(img_path):
            return img_path
    return None

def load_and_preprocess(src_path, method="contrast_focused"):
    raw_pil = Image.open(src_path).convert("RGB")
    cleaned_pil = remove_black_borders(raw_pil)
    img_np = np.array(cleaned_pil)
    
    # Apply preprocessing.py
    processed_np = preprocess_pipeline(img_np, target_size=(224, 224), method=method)
    return Image.fromarray(processed_np)

def balance_dataset(method="contrast_focused"):
    df = pd.read_csv(CSV_PATH)
    classes = df['dx'].unique()

    os.makedirs(BALANCED_DIR, exist_ok=True)

    print(f"Initiating class balancing with pre-processing '{method}'...")

    for cls in classes:
        cls_dir = os.path.join(BALANCED_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)

        cls_df = df[df['dx'] == cls]
        image_ids = cls_df['image_id'].tolist()
        num_images = len(image_ids)

        if num_images >= TARGET_COUNT:
            # Undersampling (nv class crop)
            selected_ids = cls_df.sample(n=TARGET_COUNT, random_state=42)['image_id'].tolist()
            for img_id in tqdm(selected_ids, desc=f"Cutting {cls}"):
                src_path = find_image_path(img_id)
                if src_path:
                    img_prep = load_and_preprocess(src_path, method=method)
                    img_prep.save(os.path.join(cls_dir, f"{img_id}.jpg"))

        else:
            # Oversampling + Data Augmentation
            print(f"\nAugmenting class {cls} ({num_images} -> {TARGET_COUNT})...")

            # Save adjusted originals
            for img_id in image_ids:
                src_path = find_image_path(img_id)
                if src_path:
                    img_prep = load_and_preprocess(src_path, method=method)
                    img_prep.save(os.path.join(cls_dir, f"{img_id}.jpg"))

            # Generate augmented images
            aug_needed = TARGET_COUNT - num_images
            i = 0
            pbar = tqdm(total=aug_needed, desc=f"Augmenting {cls}")

            while i < aug_needed:
                for img_id in image_ids:
                    if i >= aug_needed:
                        break

                    src_path = find_image_path(img_id)
                    if src_path:
                        img_prep = load_and_preprocess(src_path, method=method)
                        
                        # Generate a 100% clean, upscaled image with no black borders
                        aug_img = apply_augmentation(img_prep)

                        out_path = os.path.join(cls_dir, f"{img_id}_aug_{i}.jpg")
                        aug_img.save(out_path)
                        i += 1
                        pbar.update(1)
            pbar.close()

    print(f"\n Balanced Dataset saved in: {BALANCED_DIR}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balancing the HAM10000 dataset with pre-processing.")
    parser.add_argument(
        "--method",
        type=str,
        default="contrast_focused",
        choices=["baseline", "hair_removal", "color_constancy", "contrast_focused"],
        help="Preprocessing pipeline to be applied."
    )
    args = parser.parse_args()

    balance_dataset(method=args.method)