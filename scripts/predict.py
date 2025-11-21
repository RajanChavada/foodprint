from ultralytics import YOLO
import os
from pathlib import Path
import random
import json

def load_nutrition_db():
    try:
        with open('data/nutrition.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: data/nutrition.json not found. Macro estimation disabled.")
        return {}

def predict():
    # Load nutrition database
    nutrition_db = load_nutrition_db()

    # Load the trained model
    # Note: The path might vary slightly depending on the run name increment (foodprint_v1, foodprint_v12, etc.)
    # We'll try to find the latest run or default to foodprint_v1
    model_path = Path('models/foodprint_v1/weights/best.pt')
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        # Try to find any best.pt in models/
        found_models = list(Path('models').glob('**/weights/best.pt'))
        if found_models:
            model_path = found_models[-1] # Take the last one (likely most recent)
            print(f"Found model at: {model_path}")
        else:
            return

    model = YOLO(str(model_path))

    # Get a random image from the test set
    test_images_dir = Path('data/processed/test/images')
    if not test_images_dir.exists():
        print("Test directory not found.")
        return

    test_images = list(test_images_dir.glob('*'))
    if not test_images:
        print("No images found in test directory.")
        return

    # Pick a few random images
    selected_images = random.sample(test_images, min(3, len(test_images)))

    print(f"Running inference on {len(selected_images)} images...")
    
    results = model.predict(selected_images, save=True, project='runs/predict', name='test_inference', exist_ok=True)
    
    for i, r in enumerate(results):
        print(f"\nImage: {selected_images[i].name}")
        
        detected_counts = {}
        
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            
            # Only count confident detections
            if conf > 0.4:
                detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
                print(f"  Detected: {cls_name} (Confidence: {conf:.2f})")
        
        # Calculate Macros
        if detected_counts:
            print("\n  --- Estimated Nutrition ---")
            total_cals = 0
            total_protein = 0
            total_carbs = 0
            total_fat = 0
            
            for food, count in detected_counts.items():
                if food in nutrition_db:
                    info = nutrition_db[food]
                    cals = info['calories'] * count
                    prot = info['protein'] * count
                    carbs = info['carbs'] * count
                    fat = info['fat'] * count
                    
                    total_cals += cals
                    total_protein += prot
                    total_carbs += carbs
                    total_fat += fat
                    
                    print(f"  {count}x {food} ({info['unit']}): {cals} kcal")
                else:
                    print(f"  {count}x {food}: No nutrition info found")
            
            print(f"  TOTAL: {total_cals} kcal | P: {total_protein}g | C: {total_carbs}g | F: {total_fat}g")
        else:
            print("  No confident detections found.")

    print(f"\nResults saved to runs/predict/test_inference")

if __name__ == "__main__":
    predict()
