"""
Create Image Path Mappings for MLE Ensemble Testing
====================================================

This script maps PlantVillage images to the test/validation/train sets
based on disease labels from the preprocessed text data.

Author: NLP Project Team
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Paths
BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
IMAGE_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "Image Dataset", "PlantVillage")
PREPROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "image_mappings.pkl")

def normalize_label(label):
    """
    Normalize disease label to match folder names
    
    Examples:
        "Bacterial spot" -> "Bacterial_spot"
        "Tomato__Tomato_mosaic_virus" -> "Tomato__Tomato_mosaic_virus"
    """
    # Replace spaces with underscores
    normalized = label.replace(" ", "_")
    # Remove extra characters
    normalized = normalized.replace("__", "_")
    return normalized

def create_label_to_folder_mapping():
    """
    Create mapping between disease labels and PlantVillage folder names
    
    Returns:
        dict: Mapping of normalized labels to folder paths
    """
    print("\n" + "="*80)
    print("CREATING LABEL TO FOLDER MAPPING")
    print("="*80)
    
    # Get all folders in PlantVillage directory
    folders = [f for f in os.listdir(IMAGE_DATASET_DIR) 
               if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, f))]
    
    print(f"\n📁 Found {len(folders)} disease folders:")
    for i, folder in enumerate(sorted(folders), 1):
        num_images = len(list(Path(os.path.join(IMAGE_DATASET_DIR, folder)).glob("*.jpg"))) + \
                     len(list(Path(os.path.join(IMAGE_DATASET_DIR, folder)).glob("*.JPG"))) + \
                     len(list(Path(os.path.join(IMAGE_DATASET_DIR, folder)).glob("*.png")))
        print(f"   {i:2d}. {folder:50s} ({num_images:4d} images)")
    
    # Create mapping
    label_to_folder = {}
    for folder in folders:
        # Store both the original and normalized versions
        label_to_folder[folder] = folder
        # Also add without prefix (e.g., "Tomato_Early_blight" for easier matching)
        if "_" in folder:
            parts = folder.split("_")
            # Try variations
            label_to_folder["_".join(parts[1:])] = folder
            label_to_folder[" ".join(parts[1:])] = folder
    
    print(f"\n✅ Created {len(label_to_folder)} label mappings")
    return label_to_folder, folders

def get_images_from_folder(folder_path, num_images=None):
    """
    Get list of image paths from a folder
    
    Args:
        folder_path: Path to folder
        num_images: Number of images to sample (None for all)
        
    Returns:
        list: List of image paths
    """
    extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
    images = []
    
    for ext in extensions:
        images.extend(list(Path(folder_path).glob(ext)))
    
    # Convert to strings
    images = [str(img) for img in images]
    
    # Shuffle for randomness
    random.shuffle(images)
    
    # Sample if requested
    if num_images and len(images) > num_images:
        images = images[:num_images]
    
    return images

def match_label_to_folder(label, label_to_folder, folders):
    """
    Try to match a disease label to a folder name
    
    Args:
        label: Disease label from text data
        label_to_folder: Mapping dictionary
        folders: List of available folders
        
    Returns:
        str: Matched folder name or None
    """
    # Try exact match
    if label in label_to_folder:
        return label_to_folder[label]
    
    # Try normalized version
    normalized = normalize_label(label)
    if normalized in label_to_folder:
        return label_to_folder[normalized]
    
    # Try partial matching (find folder containing label keywords)
    label_lower = label.lower().replace(" ", "_")
    for folder in folders:
        folder_lower = folder.lower()
        if label_lower in folder_lower or folder_lower in label_lower:
            return folder
    
    # Try without plant prefix
    if "_" in label:
        parts = label.split("_")
        for i in range(len(parts)):
            candidate = "_".join(parts[i:])
            if candidate in label_to_folder:
                return label_to_folder[candidate]
    
    return None

def create_image_splits(preprocessed_data, label_to_folder, folders):
    """
    Create train/val/test image splits matching the text data splits
    
    Args:
        preprocessed_data: Dictionary containing preprocessed text data
        label_to_folder: Mapping of labels to folders
        folders: List of available folders
        
    Returns:
        dict: Image paths for train/val/test sets
    """
    print("\n" + "="*80)
    print("CREATING IMAGE SPLITS")
    print("="*80)
    
    label_encoder = preprocessed_data['label_encoder']
    
    # Get labels for each split
    y_train = preprocessed_data['y_train']
    y_val = preprocessed_data['y_val']
    y_test = preprocessed_data['y_test']
    
    splits = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    
    image_mappings = {
        'image_paths_train': [],
        'image_paths_val': [],
        'image_paths_test': [],
        'y_train_for_images': y_train,
        'y_val_for_images': y_val,
        'y_test_for_images': y_test,
    }
    
    # Statistics
    stats = defaultdict(int)
    unmatched_labels = set()
    
    for split_name, y_split in splits.items():
        print(f"\n📊 Processing {split_name} split ({len(y_split)} samples)...")
        
        image_paths = []
        
        # Group by class to distribute images evenly
        class_indices = defaultdict(list)
        for idx, label_idx in enumerate(y_split):
            class_indices[label_idx].append(idx)
        
        # Pre-load available images for each class
        class_image_pools = {}
        
        for label_idx in class_indices.keys():
            label_name = label_encoder.classes_[label_idx]
            matched_folder = match_label_to_folder(label_name, label_to_folder, folders)
            
            if matched_folder:
                folder_path = os.path.join(IMAGE_DATASET_DIR, matched_folder)
                available_images = get_images_from_folder(folder_path)
                class_image_pools[label_idx] = available_images
                stats[f"{split_name}_matched"] += len(class_indices[label_idx])
            else:
                class_image_pools[label_idx] = []
                stats[f"{split_name}_unmatched"] += len(class_indices[label_idx])
                unmatched_labels.add(label_name)
        
        # Assign images to each sample
        for idx, label_idx in enumerate(y_split):
            if label_idx in class_image_pools and len(class_image_pools[label_idx]) > 0:
                # Cycle through available images
                img_idx = idx % len(class_image_pools[label_idx])
                image_paths.append(class_image_pools[label_idx][img_idx])
            else:
                # Use a placeholder or None
                image_paths.append(None)
        
        image_mappings[f'image_paths_{split_name}'] = image_paths
        print(f"   ✅ Mapped {stats[f'{split_name}_matched']} images")
        if stats[f'{split_name}_unmatched'] > 0:
            print(f"   ⚠️  Could not map {stats[f'{split_name}_unmatched']} samples")
    
    # Show unmatched labels
    if unmatched_labels:
        print(f"\n⚠️  Unmatched labels ({len(unmatched_labels)}):")
        for label in sorted(unmatched_labels):
            print(f"   • {label}")
    
    return image_mappings, stats

def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" "*20 + "IMAGE PATH MAPPING CREATION")
    print("="*80)
    
    # Load preprocessed data
    print(f"\n📂 Loading preprocessed data...")
    with open(PREPROCESSED_DATA_PATH, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    print(f"✅ Loaded data:")
    print(f"   • Train samples: {len(preprocessed_data['y_train']):,}")
    print(f"   • Validation samples: {len(preprocessed_data['y_val']):,}")
    print(f"   • Test samples: {len(preprocessed_data['y_test']):,}")
    print(f"   • Number of classes: {len(preprocessed_data['label_encoder'].classes_)}")
    
    # Create label to folder mapping
    label_to_folder, folders = create_label_to_folder_mapping()
    
    # Create image splits
    image_mappings, stats = create_image_splits(preprocessed_data, label_to_folder, folders)
    
    # Save mappings
    print("\n" + "="*80)
    print("SAVING IMAGE MAPPINGS")
    print("="*80)
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(image_mappings, f)
    
    print(f"\n💾 Image mappings saved: {OUTPUT_PATH}")
    print(f"   • File size: {os.path.getsize(OUTPUT_PATH) / 1024**2:.2f} MB")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_matched = sum(v for k, v in stats.items() if 'matched' in k)
    total_unmatched = sum(v for k, v in stats.items() if 'unmatched' in k)
    total = total_matched + total_unmatched
    
    print(f"\n📊 Overall Statistics:")
    print(f"   • Total samples: {total:,}")
    print(f"   • Successfully mapped: {total_matched:,} ({total_matched/total*100:.1f}%)")
    print(f"   • Unmatched: {total_unmatched:,} ({total_unmatched/total*100:.1f}%)")
    
    for split in ['train', 'val', 'test']:
        matched = stats.get(f'{split}_matched', 0)
        unmatched = stats.get(f'{split}_unmatched', 0)
        total_split = matched + unmatched
        if total_split > 0:
            print(f"\n   {split.upper()}:")
            print(f"     - Matched: {matched:,} ({matched/total_split*100:.1f}%)")
            print(f"     - Unmatched: {unmatched:,} ({unmatched/total_split*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ IMAGE MAPPING CREATION COMPLETE")
    print("="*80)
    print("\n🎯 Next step: Run ensemble evaluation script")
    print("   python scripts/22_evaluate_ensemble_mle.py")

if __name__ == "__main__":
    main()
