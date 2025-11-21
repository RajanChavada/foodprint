import os
import sys
from dotenv import load_dotenv

# Load environment variables first, before importing libraries that might check them
load_dotenv()

from roboflow import Roboflow

# Try to import kaggle, handling the case where credentials are missing
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except OSError:
    print("Warning: Kaggle credentials not found. Kaggle datasets will be skipped.")
    print("To fix: Add KAGGLE_USERNAME and KAGGLE_KEY to your .env file, or place kaggle.json in ~/.kaggle/")
    KAGGLE_AVAILABLE = False
except Exception as e:
    print(f"Warning: Failed to import kaggle: {e}")
    KAGGLE_AVAILABLE = False

def download_roboflow_datasets():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_roboflow_api_key_here":
        print("Error: ROBOFLOW_API_KEY not found in .env file.")
        return

    rf = Roboflow(api_key=api_key)

    # Dataset 1: DataCluster Labs Indian Food
    print("\nDownloading DataCluster Labs Indian Food dataset...")
    try:
        project = rf.workspace("datacluster-labs-agryi").project("indian-food-image")
        version = project.version(1)
        dataset = version.download("yolov8", location="data/raw/roboflow_dataset_1")
        print("Dataset 1 downloaded successfully.")
    except Exception as e:
        print(f"Failed to download Dataset 1: {e}")

    # Dataset 2: Smart India Hackathon
    print("\nDownloading Smart India Hackathon dataset...")
    try:
        project = rf.workspace("smart-india-hackathon").project("indian-food-yolov5")
        version = project.version(1)
        dataset = version.download("yolov8", location="data/raw/roboflow_dataset_2")
        print("Dataset 2 downloaded successfully.")
    except Exception as e:
        print(f"Failed to download Dataset 2: {e}")

def download_kaggle_datasets():
    if not KAGGLE_AVAILABLE:
        print("\nSkipping Kaggle datasets due to missing credentials.")
        return

    print("\nDownloading Kaggle datasets...")
    
    # Ensure kaggle.json is set up or env vars are present
    # We'll assume the user has set up ~/.kaggle/kaggle.json or env vars
    
    try:
        # Dataset 3: Indian Food Image Dataset
        print("Downloading dataclusterlabs/indian-food-image-dataset...")
        kaggle.api.dataset_download_files(
            'dataclusterlabs/indian-food-image-dataset',
            path='data/raw/kaggle_indian_food',
            unzip=True
        )
        print("Kaggle Dataset 1 downloaded.")
    except Exception as e:
        print(f"Failed to download Kaggle Dataset 1: {e}")

    try:
        # Dataset 4: Indian Food Dataset (Metadata)
        print("Downloading sukhmandeepsinghbrar/indian-food-dataset...")
        kaggle.api.dataset_download_files(
            'sukhmandeepsinghbrar/indian-food-dataset',
            path='data/raw/kaggle_metadata',
            unzip=True
        )
        print("Kaggle Dataset 2 downloaded.")
    except Exception as e:
        print(f"Failed to download Kaggle Dataset 2: {e}")

    # Dataset 3: Food-101 (via Kaggle)
    try:
        print("Downloading dansbecker/food-101...")
        kaggle.api.dataset_download_files(
            'dansbecker/food-101',
            path='data/raw/food-101',
            unzip=True
        )
        print("Food-101 downloaded.")
    except Exception as e:
        print(f"Failed to download Food-101: {e}")

    # Dataset 4: UEC-Food-256 (via Kaggle mirror if available, or warn)
    # Note: UEC-Food-256 is often not on Kaggle officially. 
    # We will try a known mirror or skip for now as direct URL download is complex without browser.
    try:
        print("Downloading taukah/uecfood256...")
        kaggle.api.dataset_download_files(
            'taukah/uecfood256',
            path='data/raw/uec_food_256',
            unzip=True
        )
        print("UEC-Food-256 downloaded.")
    except Exception as e:
        print(f"Failed to download UEC-Food-256 (Kaggle mirror): {e}")

def download_direct_datasets():
    """
    Download datasets from direct URLs if not available on Kaggle/Roboflow.
    """
    import requests
    import zipfile
    import tarfile
    
    print("\nChecking for direct downloads...")
    # Add direct download logic here if needed
    pass

def download_open_images_subset():
    """
    Download a subset of Open Images V7 for the classes defined in data/global_classes.txt.
    Uses fiftyone to fetch specific classes and limit samples to save space.
    """
    print("\nDownloading Open Images subset via FiftyOne...")
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("Error: fiftyone not installed. Please run: pip install fiftyone")
        return

    # Read target classes
    try:
        with open('data/global_classes.txt', 'r') as f:
            target_classes = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: data/global_classes.txt not found.")
        return

    print(f"Targeting {len(target_classes)} classes: {target_classes[:5]}...")

    # Open Images uses specific class names (Capitalized). 
    # We need to map our snake_case to Title Case (e.g., 'french_fries' -> 'French fries')
    # Ideally, we'd have a mapping, but for now let's try simple Title Case conversion.
    oi_classes = [c.replace('_', ' ').title() for c in target_classes]
    
    # Note: Open Images might not have ALL our classes exactly as named.
    # FiftyOne will warn/skip if not found.
    
    print("Downloading validation split (smaller, good for testing)...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=oi_classes,
        max_samples=50, # Limit to 50 images per class to save space
        dataset_dir="data/raw/open_images",
        seed=42,
        shuffle=True,
        only_matching=True
    )
    
    print(f"Downloaded {len(dataset)} images from Open Images.")

if __name__ == "__main__":
    print("Starting data download...")
    
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # download_roboflow_datasets() # Skip for now to focus on global
    # download_kaggle_datasets()   # Skip for now
    
    download_open_images_subset()
    
    print("\nData download process completed.")
