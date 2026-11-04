
#!/usr/bin/env python3 TRANSLATION CHECKED
# -*- coding: utf-8 -*-

"""
HAM10000 Dataset Preparation and Balancing
==========================================

This script prepares the HAM10000 dataset for skin lesion
classification while preventing data leakage.

Main steps:
1. Load HAM10000 metadata and images.
2. Split the dataset by lesion_id (70/15/15).
3. Verify that no lesion appears in multiple subsets.
4. Apply image preprocessing.
5. Optionally perform data augmentation on training data only.
6. Save training, validation, and test datasets.

By default, no augmentation is performed. This allows
controlled comparisons between models trained with and
without online data augmentation.

Usage:
    python3 balance_dataset_new.py --output-dir Datasets_safe

    python3 balance_dataset_new.py \
        --output-dir Datasets_balanced_safe \
        --balance-training
"""

import os
import argparse
import random

from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


# 1. CONFIGURATION

BASE_DIR = Path(".")

METADATA_FILE = BASE_DIR / "HAM10000_metadata.csv"

IMAGE_DIRECTORIES = [
    BASE_DIR / "HAM10000_images_part_1",
    BASE_DIR / "HAM10000_images_part_2"
]

DEFAULT_OUTPUT_DIR = Path("Datasets_safe")

DEFAULT_IMAGE_SIZE = 224
DEFAULT_SEED = 42
DEFAULT_TARGET_COUNT = 1000


# 2. FIND IMAGE PATHS


def get_image_paths(base_dir):

    """
    Find all HAM10000 images in the two original directories.

    Returns:
        Dictionary mapping image_id to image path.
    """

    image_paths = {}

    for folder_name in [
        "HAM10000_images_part_1",
        "HAM10000_images_part_2"
    ]:

        folder = base_dir / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {folder}"
            )

        for image_path in folder.glob("*.jpg"):

            image_id = image_path.stem

            if image_id in image_paths:
                raise ValueError(
                    f"Duplicate image ID found: {image_id}"
                )

            image_paths[image_id] = image_path

    return image_paths



# 3. SPLIT DATASET BY LESION ID


def split_dataset(metadata, seed=42):

    """
    Split the dataset into:
        70% training
        15% validation
        15% testing

    Splitting is performed by lesion_id instead of image_id.

    This ensures that images belonging to the same lesion
    cannot appear in different subsets.
    """

    print("\nSplitting dataset by lesion_id...")

    # First split: training (70%) and remaining (30%)

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed
    )

    train_idx, remaining_idx = next(
        first_split.split(
            metadata,
            groups=metadata["lesion_id"]
        )
    )

    train_df = metadata.iloc[train_idx].copy()

    remaining_df = metadata.iloc[remaining_idx].copy()

    # Second split: validation (15%) and testing (15%)

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed + 1
    )

    val_idx, test_idx = next(
        second_split.split(
            remaining_df,
            groups=remaining_df["lesion_id"]
        )
    )

    val_df = remaining_df.iloc[val_idx].copy()

    test_df = remaining_df.iloc[test_idx].copy()

    splits = {
        "training": train_df,
        "validation": val_df,
        "test": test_df
    }

    # Verify that no lesion is shared between subsets

    for split_a, split_b in [
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test")
    ]:

        lesions_a = set(splits[split_a]["lesion_id"])
        lesions_b = set(splits[split_b]["lesion_id"])

        overlap = lesions_a.intersection(lesions_b)

        if overlap:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED: "
                f"{len(overlap)} lesions shared between "
                f"{split_a} and {split_b}"
            )

    print("Dataset split completed successfully.")
    print("No overlapping lesion IDs detected.")

    return splits



# 4. IMAGE PREPROCESSING


def preprocess_image(
    image_path,
    image_size=224,
    method="baseline"
):

    """
    Load and preprocess an image.

    Available methods:
        baseline
        hair_removal
        color_constancy
        contrast_focused

    The baseline method only resizes the image.

    Other methods use the existing preprocessing.py module.
    """

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    if method == "baseline":

        return image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR
        )

    from preprocessing import preprocess_pipeline

    image_array = np.asarray(image)

    processed_array = preprocess_pipeline(
        image_array,
        target_size=(image_size, image_size),
        method=method
    )

    return Image.fromarray(
        np.asarray(processed_array, dtype=np.uint8)
    ).convert("RGB")



# 5. DATA AUGMENTATION


def apply_augmentation(image, rng):

    """
    Apply random transformations to a training image.

    Transformations:
        Random rotation: 0, 90, 180, or 270 degrees
        Random horizontal flip
        Random vertical flip

    Augmentation is never applied to validation or test data.
    """

    # Random rotation

    angle = rng.choice([0, 90, 180, 270])

    image = image.rotate(
        angle,
        expand=True
    )

    # Random horizontal flip

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    # Random vertical flip

    if rng.random() < 0.5:
        image = ImageOps.flip(image)

    return image



# 6. SAVE ORIGINAL IMAGES


def save_split(
    dataframe,
    split_name,
    output_dir,
    image_size,
    method
):

    """
    Save the original images of one dataset subset.

    Output structure:
        output_dir/split_name/class_name/image_id.jpg
    """

    print(f"\nProcessing {split_name} images...")

    for row in tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc=split_name
    ):

        class_directory = (
            output_dir / split_name / row.dx
        )

        class_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        image = preprocess_image(
            row.path,
            image_size=image_size,
            method=method
        )

        output_path = (
            class_directory / f"{row.image_id}.jpg"
        )

        image.save(
            output_path,
            quality=95
        )



# 7. BALANCE TRAINING DATA ONLY


def balance_training_dataset(
    train_df,
    output_dir,
    image_size=224,
    target_count=1000,
    seed=42
):

    """
    Generate augmented images for minority classes.

    IMPORTANT:
        Only the training set is modified.

        Validation and test sets remain unchanged.

    Classes with fewer than target_count images receive
    additional augmented images.

    Classes with more than target_count images retain
    all their original images.
    """

    print("\nBalancing training dataset...")

    rng = random.Random(seed)

    for class_name, class_df in train_df.groupby("dx"):

        class_directory = (
            output_dir / "training" / class_name
        )

        original_paths = [
            class_directory / f"{image_id}.jpg"
            for image_id in class_df["image_id"]
        ]

        original_paths = [
            path
            for path in original_paths
            if path.exists()
        ]

        original_count = len(original_paths)

        if original_count == 0:
            raise ValueError(
                f"No training images found for {class_name}"
            )

        augmentation_needed = max(
            0,
            target_count - original_count
        )

        print(
            f"\nClass: {class_name}"
            f"\nOriginal images: {original_count}"
            f"\nAugmented images required: {augmentation_needed}"
        )

        for i in tqdm(
            range(augmentation_needed),
            desc=f"Augmenting {class_name}"
        ):

            source_path = rng.choice(original_paths)

            with Image.open(source_path) as image_file:
                image = image_file.convert("RGB")

                augmented_image = apply_augmentation(
                    image,
                    rng
                )

            augmented_image = augmented_image.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR
            )

            output_filename = (
                f"{source_path.stem}_aug_{i:05d}.jpg"
            )

            augmented_image.save(
                class_directory / output_filename,
                quality=95
            )

    print("\nTraining augmentation completed.")



# 8. MAIN FUNCTION


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the HAM10000 dataset using "
            "lesion-level splitting without data leakage."
        )
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory containing the original HAM10000 files."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the prepared dataset will be saved."
    )

    parser.add_argument(
        "--method",
        choices=[
            "baseline",
            "hair_removal",
            "color_constancy",
            "contrast_focused"
        ],
        default="baseline",
        help="Image preprocessing method."
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Output image width and height."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--balance-training",
        action="store_true",
        help="Generate augmented images for minority training classes."
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Minimum number of training images per class."
    )

    args = parser.parse_args()

    # Prevent accidental mixing of different experiments

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        parser.error(
            "Output directory is not empty. "
            "Choose a new directory to avoid mixing experiments."
        )

    if args.img_size < 1 or args.target_count < 1:
        parser.error(
            "Image size and target count must be positive."
        )

    # Load metadata

    metadata_path = (
        args.base_dir / "HAM10000_metadata.csv"
    )

    print("\nLoading HAM10000 metadata...")

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "image_id",
        "lesion_id",
        "dx"
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    if metadata[list(required_columns)].isna().any().any():
        raise ValueError(
            "Metadata contains missing image, lesion, or class IDs."
        )

    if metadata["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image IDs found in metadata."
        )

    # Locate original images

    image_paths = get_image_paths(args.base_dir)

    metadata["path"] = metadata["image_id"].map(
        image_paths
    )

    missing_images = metadata["path"].isna().sum()

    if missing_images > 0:
        raise FileNotFoundError(
            f"{missing_images} images listed in the "
            "metadata were not found."
        )

    print(
        f"Total images: {len(metadata)}"
    )

    print(
        f"Unique lesions: {metadata['lesion_id'].nunique()}"
    )

    # Split dataset before applying augmentation

    datasets = split_dataset(
        metadata,
        seed=args.seed
    )

    # Save each dataset subset

    for split_name, split_df in datasets.items():

        save_split(
            split_df,
            split_name,
            args.output_dir,
            args.img_size,
            args.method
        )

        # Save metadata for reproducibility

        split_df[
            ["image_id", "lesion_id", "dx"]
        ].to_csv(
            args.output_dir / f"{split_name}_metadata.csv",
            index=False
        )

        print(
            f"{split_name}: "
            f"{len(split_df)} images, "
            f"{split_df['lesion_id'].nunique()} unique lesions"
        )

    # Optional augmentation of training data

    if args.balance_training:

        balance_training_dataset(
            datasets["training"],
            args.output_dir,
            image_size=args.img_size,
            target_count=args.target_count,
            seed=args.seed
        )

    print("\n")
    print("DATASET PREPARATION COMPLETED")
    print("\n")

    print("No lesion-level data leakage detected.")

    print(
        f"Dataset saved to: "
        f"{args.output_dir.resolve()}"
    )

    print(
        "\nNote: validation and test images "
        "were not augmented."
    )



# 9. EXECUTION


if __name__ == "__main__":
    main()
