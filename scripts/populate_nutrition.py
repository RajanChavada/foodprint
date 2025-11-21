import json
import random

def populate_nutrition():
    # Load class list
    try:
        with open('data/global_classes.txt', 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: data/global_classes.txt not found.")
        return

    nutrition_db = {}
    
    # Basic default values for different categories (rough estimates)
    defaults = {
        'fruit': {'unit': 'medium', 'calories': 80, 'protein': 1, 'carbs': 20, 'fat': 0},
        'vegetable': {'unit': 'medium', 'calories': 30, 'protein': 2, 'carbs': 5, 'fat': 0},
        'meat': {'unit': 'serving (100g)', 'calories': 250, 'protein': 25, 'carbs': 0, 'fat': 15},
        'carb': {'unit': 'serving', 'calories': 200, 'protein': 5, 'carbs': 40, 'fat': 2},
        'dairy': {'unit': 'serving', 'calories': 150, 'protein': 8, 'carbs': 12, 'fat': 8},
        'sweet': {'unit': 'piece', 'calories': 250, 'protein': 3, 'carbs': 35, 'fat': 12},
        'meal': {'unit': 'serving', 'calories': 500, 'protein': 20, 'carbs': 50, 'fat': 20},
    }

    # Heuristic mapping (this would ideally be replaced by a real API lookup)
    for cls in classes:
        category = 'meal' # Default
        if cls in ['apple', 'banana', 'orange', 'grape', 'strawberry', 'watermelon', 'pineapple', 'mango', 'lemon']:
            category = 'fruit'
        elif cls in ['carrot', 'cucumber', 'lettuce', 'tomato', 'onion', 'bell_pepper', 'broccoli', 'spinach', 'mushroom', 'zucchini', 'eggplant', 'cabbage', 'cauliflower', 'green_bean', 'pea', 'asparagus', 'celery']:
            category = 'vegetable'
        elif cls in ['steak', 'chicken_breast', 'chicken_leg', 'fish', 'salmon', 'tuna', 'shrimp', 'egg', 'tofu', 'bacon', 'sausage', 'ham', 'pork_chop', 'lamb_chop']:
            category = 'meat'
        elif cls in ['rice', 'pasta', 'spaghetti', 'bread', 'bagel', 'croissant', 'potato', 'sweet_potato', 'corn', 'oatmeal']:
            category = 'carb'
        elif cls in ['cheese', 'yogurt', 'milk', 'butter']:
            category = 'dairy'
        elif cls in ['cake', 'cookie', 'donut', 'ice_cream', 'chocolate', 'muffin']:
            category = 'sweet'
        
        # Add some random variation so it looks realistic for now
        base = defaults[category]
        nutrition_db[cls] = {
            'unit': base['unit'],
            'calories': base['calories'],
            'protein': base['protein'],
            'carbs': base['carbs'],
            'fat': base['fat']
        }

    # Save to JSON
    with open('data/nutrition.json', 'w') as f:
        json.dump(nutrition_db, f, indent=4)
    
    print(f"Generated nutrition database for {len(classes)} items at data/nutrition.json")

if __name__ == "__main__":
    populate_nutrition()
