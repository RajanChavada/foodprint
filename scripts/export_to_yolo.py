import fiftyone as fo
import fiftyone.zoo as foz
import os

def export_yolo():
    print("Loading Open Images subset from FiftyOne...")
    
    # We need to re-load the dataset. 
    # Since we used load_zoo_dataset with specific classes, we should use the same args to get the view/subset.
    # Or, if it was saved as a persistent dataset, we can load it by name.
    # load_zoo_dataset usually creates a dataset named "open-images-v7-validation" if not specified.
    
    # Let's try to load the persistent one first
    try:
        dataset = fo.load_dataset("open-images-v7-validation")
        print("Loaded existing dataset 'open-images-v7-validation'")
    except ValueError:
        print("Dataset not found in DB, re-loading via zoo (should be cached)...")
        # Re-construct the class list
        with open('data/global_classes.txt', 'r') as f:
            target_classes = [line.strip() for line in f if line.strip()]
        oi_classes = [c.replace('_', ' ').title() for c in target_classes]
        
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],
            classes=oi_classes,
            max_samples=50,
            dataset_dir="data/raw/open_images",
            seed=42,
            shuffle=True,
            only_matching=True
        )

    print(f"Exporting {len(dataset)} samples to YOLO format...")
    
    # Define the export directory
    export_dir = "data/processed/global_food"
    
    # The classes list for YOLO (snake_case)
    # We need to map the Open Images classes (Title Case) back to our snake_case classes
    # or just use the snake_case list as the master list.
    
    with open('data/global_classes.txt', 'r') as f:
        yolo_classes = [line.strip() for line in f if line.strip()]
        
    # FiftyOne exports using the classes present in the dataset.
    # We want to force the order to match our global_classes.txt
    
    # However, the labels in the dataset are Title Case (e.g. "Ice Cream").
    # We need to map them or just use the Title Case labels in data.yaml and map them during inference.
    # Let's stick to the dataset's labels for the export to avoid mismatch, 
    # but we can provide the `classes` argument to filter/order them.
    
    # Wait, if we want "french_fries" in our system, but Open Images has "French Fries",
    # we should probably just use "French Fries" in our model and map it later?
    # Or we can rename the classes during export?
    # FiftyOne supports mapping.
    
    # Let's try to map them to snake_case for consistency.
    mapping = {c.replace('_', ' ').title(): c for c in yolo_classes}
    
    # Some might not match exactly (e.g. "Hot Dog" vs "Hot dog").
    # Let's inspect a few labels if possible, but for now let's just export as is 
    # and we'll generate data.yaml from the exported classes.
    
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train", # Export everything to a 'train' folder for now, we can split later or just use it
        classes=list(mapping.keys()), # Only export the classes we care about
    )
    
    # Now we need to fix the data.yaml to use our snake_case names?
    # Or just update data.yaml manually.
    
    print(f"Export complete to {export_dir}")

if __name__ == "__main__":
    export_yolo()
