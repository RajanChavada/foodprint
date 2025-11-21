import fiftyone as fo
import fiftyone.zoo as foz

print("FiftyOne version:", fo.__version__)

try:
    print("Attempting to download 'Apple'...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Apple"],
        max_samples=10,
        dataset_dir="data/raw/debug_open_images",
        seed=42,
        shuffle=True,
        only_matching=True
    )
    print("Success!")
except Exception as e:
    print("Failed!")
    print(e)
    import traceback
    traceback.print_exc()
