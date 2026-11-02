"""
Preprocessing Module for Dermatoscopic Skin Lesion Analysis (HAM10000)
Based on methods from Aakaou et al. (2026): Contrast-Focused Preprocessing...
"""
# TRANSLATE CHECKED

import cv2
import numpy as np


def apply_dull_razor(img_rgb):
    """
    Morphological hair removal via Black Hat transform and Telea inpainting.
    
    Args:
        img_rgb (np.ndarray): Original image in RGB format (uint8, [0, 255]).
    Returns:
        np.ndarray: Hair-free RGB image (uint8, [0, 255]).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_clean = cv2.inpaint(img_rgb, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img_clean


def apply_color_constancy(img_rgb, p=6):
    """
    Color constancy and illumination correction via Shades of Gray (Minkowski p-norm).
    
    Args:
        img_rgb (np.ndarray): Original image in RGB format (uint8, [0, 255]).
        p (int): Minkowski norm order (typically p=6).
    Returns:
        np.ndarray: Color-normalized RGB image (uint8, [0, 255]).
    """
    img_float = img_rgb.astype(np.float32)
    illuminant = np.power(np.mean(np.power(img_float, p), axis=(0, 1)), 1.0 / p)
    norm = np.linalg.norm(illuminant)
    if norm > 0:
        illuminant /= norm
    
    scale = (1.0 / np.sqrt(3)) / (illuminant + 1e-8)
    img_corrected = np.clip(img_float * scale, 0, 255).astype(np.uint8)
    return img_corrected


def apply_clahe_lab(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization on the Luminance channel (LAB color space).
    
    Args:
        img_rgb (np.ndarray): Image in RGB format (uint8, [0, 255]).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Grid size for local histogram equalization.
    Returns:
        np.ndarray: Contrast-enhanced RGB image (uint8, [0, 255]).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    
    lab_merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)


def apply_contrast_focused(img_rgb):
    """
    Contrast-focused pipeline combining DullRazor hair removal and CLAHE on the L channel.
    Top-performing pipeline in Aakaou et al. (2026).
    
    Args:
        img_rgb (np.ndarray): Original image in RGB format (uint8, [0, 255]).
    Returns:
        np.ndarray: Processed image in RGB format (uint8, [0, 255]).
    """
    no_hair = apply_dull_razor(img_rgb)
    contrast_img = apply_clahe_lab(no_hair, clip_limit=2.0, tile_grid_size=(8, 8))
    return contrast_img


def preprocess_pipeline(img_rgb, target_size=(224, 224), method="baseline"):
    """
    Applies the chosen preprocessing pipeline and resizes the image.
    
    Args:
        img_rgb (np.ndarray): Image in RGB format.
        target_size (tuple): Output dimensions (width, height).
        method (str): Pipeline selector.
            Options: 'baseline', 'hair_removal', 'color_constancy', 'contrast_focused'.
    Returns:
        np.ndarray: Resized preprocessed RGB image.
    """
    if method == "baseline":
        processed = img_rgb
    elif method == "hair_removal":
        processed = apply_dull_razor(img_rgb)
    elif method == "color_constancy":
        processed = apply_color_constancy(img_rgb)
    elif method == "contrast_focused":
        processed = apply_contrast_focused(img_rgb)
    else:
        raise ValueError(f"Unknown method '{method}'. Valid options are: 'baseline', 'hair_removal', 'color_constancy', 'contrast_focused'.")
    
    resized = cv2.resize(processed, target_size, interpolation=cv2.INTER_AREA)
    return resized