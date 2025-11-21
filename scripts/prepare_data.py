import os
import shutil
import yaml
import glob
from tqdm import tqdm
from pathlib import Path
import random

# Configuration
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Classes found in Roboflow Dataset 2
TARGET_CLASSES = [
    'cham_cham', 'naan', 'paneer_butter_masala', 'rasgulla'
]

def setup_directories():
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    for split in ['train', 'valid', 'test']:
        (PROCESSED_DATA_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (PROCESSED_DATA_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

def normalize_class_name(name):
    name = name.lower().strip().replace(' ', '_')
    return name

def process_roboflow_dataset(dataset_path, split_mapping={'train': 'train', 'valid': 'valid', 'test': 'test'}):
    """
    Process a Roboflow YOLOv8 dataset.
    Reads data.yaml to get class names, then copies files and remaps labels.
    """
    print(f"Processing {dataset_path}...")
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / "data.yaml"
    
    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found. Skipping.")
        return

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    source_classes = data_config.get('names', [])
    print(f"Found classes: {source_classes}")
    
    # Map source class IDs to target class IDs
    id_mapping = {}
    for idx, name in enumerate(source_classes):
        normalized = normalize_class_name(name)
        if normalized in TARGET_CLASSES:
            id_mapping[idx] = TARGET_CLASSES.index(normalized)
        else:
            print(f"Warning: Class '{name}' not in TARGET_CLASSES. Skipping.")

    for source_split, target_split in split_mapping.items():
        # Roboflow structure can vary. Sometimes it's 'train/images', sometimes just 'train'
        # We'll check for 'images' subdir
        source_split_dir = dataset_path / source_split
        source_images_dir = source_split_dir / "images"
        source_labels_dir = source_split_dir / "labels"
        
        if not source_images_dir.exists():
            # Fallback: maybe images are directly in the split folder?
            # But Roboflow standard is usually split/images
            print(f"Warning: {source_images_dir} does not exist. Skipping {source_split}.")
            continue
            
        image_files = list(source_images_dir.glob("*"))
        
        for img_path in tqdm(image_files, desc=f"Copying {source_split}"):
            label_path = source_labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
                
            # Read and remap labels
            new_lines = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                try:
                    class_id = int(parts[0])
                    if class_id in id_mapping:
                        new_class_id = id_mapping[class_id]
                        new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                except ValueError:
                    continue
            
            # Only copy if there are valid labels
            if new_lines:
                # Copy image
                shutil.copy2(img_path, PROCESSED_DATA_DIR / target_split / "images" / img_path.name)
                
                # Write new label file
                with open(PROCESSED_DATA_DIR / target_split / "labels" / label_path.name, 'w') as f:
                    f.writelines(new_lines)

def create_data_yaml():
    data = {
        'path': str(PROCESSED_DATA_DIR.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(TARGET_CLASSES),
        'names': TARGET_CLASSES
    }
    
    with open(PROCESSED_DATA_DIR / "data.yaml", 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Created data.yaml at {PROCESSED_DATA_DIR / 'data.yaml'}")

if __name__ == "__main__":
    setup_directories()
    
    # Only process the one successful dataset
    process_roboflow_dataset("data/raw/roboflow_dataset_2")
    
    create_data_yaml()
    
    # Print stats
    for split in ['train', 'valid', 'test']:
        count = len(list((PROCESSED_DATA_DIR / split / 'images').glob('*')))
        print(f"{split}: {count} images")
