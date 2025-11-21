<project_specification>
  <title>Universal Food Macro Detector & Nutrition Estimator</title>
  <objective>
    Build an end-to-end computer vision pipeline that:
      1. Accepts any food photo as input,
      2. Detects and classifies all food items present (snack, meal, fast food, produce, global cuisine),
      3. Estimates nutrition for each detected item (calories, protein, carbs, fat, serving size),
      4. Returns a summary with class names, confidence scores, bounding boxes, and macro breakdown,
      5. Is not niche to Indian or any single regional cuisine.
  </objective>
  <motivation>
    Original prototype covered only Indian cuisine due to limited training data and narrowly scoped prompt. 
    User stories require the tool to detect ANY food, including fruits, vegetables, Western/Asian foods, desserts, and fast food.
  </motivation>
  <functional_requirements>
    <fr>
      <category>Global Food Classification</category>
      <description>
        The model must recognize and distinguish a wide range of food categories including:
          - Common produce: apple, carrot, lettuce, tomato, banana, cucumber, orange, etc.
          - Protein sources: steak, eggs, chicken breast, fish, tofu, beans, paneer, etc.
          - Bakery/pastries: bread, croissant, bagel, muffin, cake, etc.
          - Dairy: milk, cheese, yogurt, butter, etc.
          - Fast food: pizza, burger, fries, chicken nuggets, hotdog, etc.
          - World cuisine/plates: sushi, pasta, fried rice, tacos, curry, lasagna, shawarma, etc.
          - Desserts & snacks: ice cream, chocolate, chips, donuts, etc.
        Include at least 100 classes. Use datasets with English labels and global coverage.
      </description>
    </fr>
    <fr>
      <category>Universal Nutrition Estimation</category>
      <description>
        For any detected food item, estimate nutrients using a lookup on global food/nutrition databases:
          - Use FoodData Central (USDA), Food-101 class-to-nutrition mapping, and other public APIs.
          - For produce, map label like "apple" or "carrot" to USDA entries.
          - For packaged/fast food, use closest generic equivalent (e.g., "French fries", "hamburger", "pizza").
          - If food is unknown, provide a macro estimate based on semantic similarity.
        Design for easy DB expansion; all detected foods, from apple to biryani, must be supported.
      </description>
    </fr>
    <fr>
      <category>Model Selection and Training Datasets</category>
      <description>
        Use or fine-tune a model on a combination of these datasets:
          - Food-101: 101K images, 101 most common foods worldwide.
          - UEC-FOOD256: 31K images, 256 classes, includes Japanese, Asian, Western, and vegetables/fruits.
          - Open Images - Food subset: broad objects, hundreds of foods, bounding boxes for many food groups and types.
          - Roboflow or Kaggle general food datasets.
        If using existing pre-trained food classification models, support open-vocabulary detection or simple transfer learning to cover more generic classes.
        Annotate or pseudo-label images to add bounding boxes to classification datasets if YOLO is used.
      </description>
    </fr>
    <fr>
      <category>Model/Inference Improvements</category>
      <description>
        Build an inference script (`predict.py`) that can:
          - Accept any JPG/PNG image,
          - Output all detected food items (class, bbox, confidence),
          - For each, output nutrition estimation,
          - Save visualizations of predictions,
          - (Optional) Allow fallback to a text classifier if image not recognized as food.
      </description>
    </fr>
    <fr>
      <category>Workflow/Code Organization</category>
      <description>
        Organize data, code, and outputs for reproducibility:
          - data/raw: store datasets (Food-101, UEC-256, Open Images subset, etc.)
          - models/: store all trained weights (not just Indian food models)
          - scripts/: for all ETL, dataset preparation, and training
          - inference/: contain robust, modular `predict.py`
          - nutrition/: global nutrition lookup tables and DB/API code
          - README.md: must state clearly that the system is for global foods, not Indian-only
      </description>
    </fr>
  </functional_requirements>
  <non_functional_requirements>
    <nfr>
      <description>
        - Must support at least 100 food classes with ≥65% mAP@0.5 on a representative test set (Food-101, Open Images val, etc.)
        - Inference time: <5s on CPU for 1 image, <2s on GPU.
        - Nutrition lookup DB/API must resolve US/UK/International English food names.
        - All model code and nutrition mapping scripts must be open source or documented for public use.
      </description>
    </nfr>
  </non_functional_requirements>
  <open_source_resources>
    <resources>
      <dataset>Food-101</dataset>
      <dataset>UEC-FOOD256</dataset>
      <dataset>Open Images V6 Food subset</dataset>
      <dataset>Kaggle/Roboflow generic food datasets</dataset>
      <api>USDA FoodData Central</api>
      <api>Nutritionix Public API</api>
      <model>YOLOv8 (if you want object detection + classification)</model>
      <model>ConvNext/EfficientNet/Food-101 transfer learning (if image-level classification only)</model>
    </resources>
  </open_source_resources>
  <success_criteria>
    <criteria>
      - Given images of an apple, a salad bowl, a burger, a Japanese lunch box, or an Indian thali, system identifies each food item (per class list), draws bounding boxes, outputs class names & confidence.
      - Each class prediction maps to a nutrition value, enabling calculation of macros for any subset of known foods.
      - README, API, and output clearly state global support, NOT regional or one-cuisine-only.
    </criteria>
  </success_criteria>
  <user_story>
    As a user, I want to upload any picture of my meal (fruit, vegetable, Western, Asian, Indian, vegan, dessert, or fast food) and get an accurate list of what foods are present and a reasonable nutrition estimate for each.
  </user_story>
  <instructions>
    When training, use Food-101 and UEC-256 as your primary datasets; add any other available general/global food datasets for more classes.
    Annotate missing bounding boxes in Food-101 (auto or manual).
    During nutrition lookup, always prefer generic/global databases and English-language mappings.
    Never optimize only for Indian food or use an Indian-only label list at any step—support general food recognition.
  </instructions>
</project_specification>
