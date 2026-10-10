import os
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

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
    Detecta y recorta automáticamente las bandas o bordes negros laterales/superiores.
    """
    img_np = np.array(img_pil)
    
    # Mascara de píxeles que no son completamente negros (umbral de tolerancia = 10)
    non_black_mask = np.any(img_np > 10, axis=2)
    
    # Encontrar las coordenadas extremas con información de imagen
    rows = np.any(non_black_mask, axis=1)
    cols = np.any(non_black_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img_pil # Si falla la detección, retorna la imagen original
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Recortar la bounding box limpia de la imagen
    cropped_np = img_np[rmin:rmax+1, cmin:cmax+1]
    return Image.fromarray(cropped_np)

def apply_augmentation(img):
    """
    Aplica rotaciones ortogonales (90°, 180°, 270°) y remueve estrictamente 
    cualquier banda negra producida por la relación de aspecto rectangular.
    """
    # 1. Elegir un ángulo ortogonal
    angle = random.choice([0, 90, 180, 270])
    
    # 2. Rotar la imagen (expand=True ajusta el lienzo al nuevo alto/ancho)
    if angle != 0:
        img = TF.rotate(img, angle=angle, expand=True)
    
    # 3. Eliminar activamente los bordes negros si existen
    img = remove_black_borders(img)
    
    # 4. Inversiones y ajustes de color sin deformaciones
    transforms_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        # Redimensionado final uniforme (ej. 224x224 para la red neuronal)
        transforms.Resize((224, 224), interpolation=TF.InterpolationMode.BILINEAR)
    ])
    
    return transforms_pipeline(img)

def find_image_path(image_id):
    for folder in ORIGINAL_IMAGES_DIRS:
        img_path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(img_path):
            return img_path
    return None

def balance_dataset():
    df = pd.read_csv(CSV_PATH)
    classes = df['dx'].unique()

    os.makedirs(BALANCED_DIR, exist_ok=True)

    print("Iniciando balanceo de clases...")

    for cls in classes:
        cls_dir = os.path.join(BALANCED_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)

        cls_df = df[df['dx'] == cls]
        image_ids = cls_df['image_id'].tolist()
        num_images = len(image_ids)

        if num_images >= TARGET_COUNT:
            # Undersampling (recorte de la clase nv)
            selected_ids = cls_df.sample(n=TARGET_COUNT, random_state=42)['image_id'].tolist()
            for img_id in tqdm(selected_ids, desc=f"Recortando {cls}"):
                src_path = find_image_path(img_id)
                if src_path:
                    # Redimensionar también las originales al tamaño estándar 224x224
                    img = Image.open(src_path).convert("RGB")
                    img_resized = img.resize((224, 224), Image.BILINEAR)
                    img_resized.save(os.path.join(cls_dir, f"{img_id}.jpg"))

        else:
            # Oversampling + Data Augmentation
            print(f"\nAumentando clase {cls} ({num_images} -> {TARGET_COUNT})...")

            # Guardar originales ajustadas
            for img_id in image_ids:
                src_path = find_image_path(img_id)
                if src_path:
                    img = Image.open(src_path).convert("RGB")
                    img_resized = img.resize((224, 224), Image.BILINEAR)
                    img_resized.save(os.path.join(cls_dir, f"{img_id}.jpg"))

            # Generar imágenes aumentadas
            aug_needed = TARGET_COUNT - num_images
            i = 0
            pbar = tqdm(total=aug_needed, desc=f"Augmenting {cls}")

            while i < aug_needed:
                for img_id in image_ids:
                    if i >= aug_needed:
                        break

                    src_path = find_image_path(img_id)
                    if src_path:
                        img = Image.open(src_path).convert("RGB")
                        
                        # Generar aumentada 100% limpia sin bordes negros
                        aug_img = apply_augmentation(img)

                        out_path = os.path.join(cls_dir, f"{img_id}_aug_{i}.jpg")
                        aug_img.save(out_path)
                        i += 1
                        pbar.update(1)
            pbar.close()

    print(f"\n¡Dataset balanceado guardado en: {BALANCED_DIR}!")

if __name__ == "__main__":
    balance_dataset()