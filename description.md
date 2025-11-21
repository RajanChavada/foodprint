<project_specification>
  <metadata>
    <project_name>Foodprint: Generic Food & Indian Cuisine Macro Tracker</project_name>
    <objective>Build a comprehensive computer vision application that detects a wide variety of foods (focusing on Indian cuisine + common global dishes) and estimates nutritional macros.</objective>
    <development_approach>Rapid MVP iteration with Google Gemini Code Assist</development_approach>
    <timeline>7 days to working prototype</timeline>
    <developer_profile>
      <background>Software engineering student with ML/AI experience</background>
      <expertise>LLM fine-tuning, Hugging Face models, full-stack development</expertise>
      <current_focus>Learning computer vision, object detection, production ML deployment</current_focus>
      <location>Toronto, Canada (access to diverse Indian/Gujarati restaurants)</location>
    </developer_profile>
  </metadata>

  <!-- ============================================ -->
  <!-- FUNCTIONAL REQUIREMENTS -->
  <!-- ============================================ -->
  
  <functional_requirements>
    
    <requirement id="FR-1" priority="P0_CRITICAL">
      <category>Food Detection & Classification</category>
      <description>Detect and classify food items from uploaded images using computer vision</description>
      
      <sub_requirements>
        <sub_req id="FR-1.1">
          <title>Image Input Processing</title>
          <acceptance_criteria>
            - Accept JPEG/PNG images up to 10MB file size
            - Support resolutions from 400x400 to 4000x4000 pixels
            - Preprocess to 640x640 for YOLO inference
            - Convert to RGB color space if needed
            - Handle various orientations (top-down preferred)
          </acceptance_criteria>
          <technical_details>
            - Input formats: .jpg, .jpeg, .png
            - Preprocessing: PIL/OpenCV resize, normalize to [0,1]
            - Target resolution: 640x640 (YOLO standard)
          </technical_details>
        </sub_req>
        
        <sub_req id="FR-1.2">
          <title>Multi-Food Detection</title>
          <acceptance_criteria>
            - Detect up to 10 separate food items per image
            - Return bounding box coordinates [x1, y1, x2, y2] for each detection
            - Confidence score threshold: 0.5 minimum
            - Process single image in under 2 seconds on GPU
          </acceptance_criteria>
          <technical_details>
            - Model: YOLOv8 Nano (yolov8n.pt) or Small (yolov8s.pt)
            - Non-max suppression (NMS) IoU threshold: 0.45
            - Output format: List of {food_name, confidence, bbox}
          </technical_details>
        </sub_req>
        
        <sub_req id="FR-1.3">
          <title>Food Classification</title>
          <acceptance_criteria>
            - Classify 100+ common dishes (20 Indian + 80 Global)
            - Achieve 75%+ mean Average Precision (mAP@0.5) on validation set
            - Return class name and confidence score for each detection
          </acceptance_criteria>
          <supported_dishes>
            Core Indian Dishes (20):
            - Biryani, Samosa, Paneer Tikka, Butter Chicken, Dal Makhani, etc.
            
            Global/Generic Dishes (80+):
            - Pizza, Burger, Salad, Pasta, Rice, Steak, Sushi, Sandwich, etc.
            - Fruits: Apple, Banana, Orange
            - Derived from Food-101 and UEC-Food classes
          </supported_dishes>
        </sub_req>
      </sub_requirements>
    </requirement>
    
    <requirement id="FR-2" priority="P0_CRITICAL">
      <category>Macro Estimation System</category>
      <description>Estimate nutritional macros based on detected foods and portion sizes</description>
      
      <sub_requirements>
        <sub_req id="FR-2.1">
          <title>Portion Size Estimation (MVP - Simplified)</title>
          <acceptance_criteria>
            - Estimate portion size in grams with ±30% accuracy acceptable
            - Use bounding box area as proxy for portion size
            - Clamp estimates to reasonable ranges (50g-500g per item)
          </acceptance_criteria>
          <technical_details>
            MVP approach (bbox-based):
            - Formula: grams = bbox_area_pixels * pixel_to_cm_ratio² * average_depth * food_density
            - Average density assumption: 0.7 g/cm³ (configurable per food type)
          </technical_details>
        </sub_req>
        
        <sub_req id="FR-2.2">
          <title>Nutrition Data Lookup</title>
          <acceptance_criteria>
            - Fetch calories, protein, carbs, fat for each detected food
            - Use hierarchical fallback: Custom DB → USDA API → Default values
          </acceptance_criteria>
          <data_sources>
            1. Custom Database (SQLite)
               - Curated entries for supported classes
            2. USDA FoodData Central API
               - Fallback for generic lookups
            3. Hardcoded Defaults
               - Average values for broad categories (e.g., "Pizza")
          </data_sources>
        </sub_req>
        
        <sub_req id="FR-2.3">
          <title>Total Macro Calculation</title>
          <acceptance_criteria>
            - Sum macros across all detected foods in image
            - Scale individual food macros by estimated portion size
          </acceptance_criteria>
        </sub_req>
      </sub_requirements>
    </requirement>
    
    <requirement id="FR-3" priority="P0_CRITICAL">
      <category>Backend API</category>
      <description>FastAPI server for model inference and data processing</description>
      <sub_requirements>
        <sub_req id="FR-3.1">
          <title>Food Analysis Endpoint</title>
          <http_spec>
            Method: POST
            Path: /analyze-food
            Content-Type: multipart/form-data
          </http_spec>
        </sub_req>
      </sub_requirements>
    </requirement>
    
    <requirement id="FR-4" priority="P0_CRITICAL">
      <category>Data Pipeline</category>
      <description>Acquire, process, and prepare training data for model fine-tuning</description>
      
      <sub_requirements>
        <sub_req id="FR-4.1">
          <title>Dataset Acquisition</title>
          <datasets>
            <dataset id="DS-1">
              <name>Roboflow DataCluster Labs Indian Food</name>
              <url>https://universe.roboflow.com/datacluster-labs-agryi/indian-food-image</url>
              <size>5,000+ images</size>
              <format>YOLOv8</format>
            </dataset>
            
            <dataset id="DS-2">
              <name>Roboflow Smart India Hackathon</name>
              <url>https://universe.roboflow.com/smart-india-hackathon/indian-food-yolov5</url>
              <size>4,298 images</size>
              <format>YOLOv8</format>
            </dataset>
            
            <dataset id="DS-3">
              <name>Food-101 (Subset/Augmented)</name>
              <url>https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/</url>
              <size>101,000 images (101 classes)</size>
              <format>Classification (needs bbox or classifier-only use)</format>
              <note>Will use a Roboflow version with bboxes if available, or use for classification head</note>
            </dataset>
            
            <dataset id="DS-4">
              <name>UEC-Food (FoodCam)</name>
              <url>http://foodcam.mobi/dataset256.html</url>
              <size>UEC-Food-256 (256 classes)</size>
              <format>Images + BBox coordinates</format>
            </dataset>
            
            <dataset id="DS-5">
              <name>Roboflow Generic Food</name>
              <url>https://universe.roboflow.com/search?q=food</url>
              <size>Varies</size>
              <format>YOLOv8</format>
              <note>Targeting "Food Detection" datasets</note>
            </dataset>
          </datasets>
          
          <acceptance_criteria>
            - Download all 4 datasets programmatically
            - Verify file integrity (checksum if available)
            - Log download progress and completion status
            - Store in organized directory structure: ./data/raw/{dataset_name}/
          </acceptance_criteria>
        </sub_req>
        
        <sub_req id="FR-4.2">
          <title>YOLO Format Verification</title>
          <yolo_format_spec>
            Expected directory structure:
            dataset/
            ├── train/
            │   ├── images/
            │   │   ├── img_001.jpg
            │   │   └── ...
            │   └── labels/
            │       ├── img_001.txt
            │       └── ...
            ├── valid/
            │   ├── images/
            │   └── labels/
            ├── test/
            │   ├── images/
            │   └── labels/
            └── data.yaml
            
            Label file format (img_001.txt):
            # Each line: class_id center_x center_y width height
            # Coordinates normalized to [0, 1]
            0 0.5 0.5 0.8 0.6
            1 0.3 0.7 0.4 0.5
            
            data.yaml structure:
            train: ./train/images
            val: ./valid/images
            test: ./test/images
            nc: 20  # number of classes
            names: ['biryani', 'samosa', 'dhokla', ...]
          </yolo_format_spec>
          
          <acceptance_criteria>
            - Verify train/valid/test directories exist
            - Check images/ and labels/ subdirectories present
            - Validate label file format (5 values per line, normalized)
            - Confirm image-label file count matches
            - Random sample inspection (10 images) for quality
          </acceptance_criteria>
        </sub_req>
        
        <sub_req id="FR-4.3">
          <title>Dataset Consolidation</title>
          <acceptance_criteria>
            - Merge multiple datasets into unified structure
            - Resolve class name conflicts (e.g., "biryani" vs "chicken_biryani")
            - Create master data.yaml with final 20 classes
            - Split: 70% train, 20% validation, 10% test
            - No data leakage between splits (unique images only)
          </acceptance_criteria>
          <technical_details>
            - Class mapping: Manual review of all dataset classes
            - Consolidation tool: Custom Python script
            - Duplicate detection: Image hash comparison (MD5/SHA256)
            - Output: ./data/processed/ directory with clean splits
          </technical_details>
        </sub_req>
      </sub_requirements>
    </requirement>
    
    <requirement id="FR-5" priority="P0_CRITICAL">
      <category>Model Training</category>
      <description>Fine-tune YOLOv8 on Indian food dataset and export for production</description>
      
      <sub_requirements>
        <sub_req id="FR-5.1">
          <title>YOLOv8 Fine-Tuning</title>
          <training_config>
            Base Model:
              - Architecture: YOLOv8 Nano (yolov8n.pt)
              - Pretrained on: COCO dataset (80 classes, 330K images)
              - Reason: Smallest variant (6.2M parameters) optimized for mobile/edge deployment
              - Performance: ~37 FPS on CPU, 200+ FPS on GPU
            
            Hyperparameters:
              epochs: 100
              batch_size: 16  # Adjust down to 8 or 4 if GPU memory insufficient
              img_size: 640
              device: '0'  # GPU 0, or 'cpu' for CPU-only training
              
              optimizer: AdamW
              lr0: 0.01  # Initial learning rate
              lrf: 0.01  # Final learning rate (OneCycleLR)
              momentum: 0.937
              weight_decay: 0.0005
              
              warmup_epochs: 3
              warmup_momentum: 0.8
              warmup_bias_lr: 0.1
            
            Data Augmentation:
              hsv_h: 0.015    # Hue variation (±1.5%)
              hsv_s: 0.7      # Saturation variation (±70%)
              hsv_v: 0.4      # Brightness variation (±40%)
              degrees: 10     # Rotation (±10°)
              translate: 0.1  # Translation (±10%)
              scale: 0.5      # Zoom in/out (±50%)
              shear: 0.0      # No shear for food images
              perspective: 0.0
              flipud: 0.5     # Vertical flip (50% prob) - critical for top-down food photos
              fliplr: 0.5     # Horizontal flip (50% prob)
              mosaic: 1.0     # Mosaic augmentation (combines 4 images)
              mixup: 0.15     # Mixup augmentation (blends 2 images)
            
            Regularization:
              dropout: 0.0    # No dropout for small model
              label_smoothing: 0.0
            
            Loss Functions:
              box_loss: CIoU (Complete IoU)
              cls_loss: BCEWithLogitsLoss (Binary Cross Entropy)
              dfl_loss: Distribution Focal Loss
          </training_config>
          
          <acceptance_criteria>
            - Start from yolov8n.pt pretrained weights
            - Train for minimum 100 epochs (early stopping if no improvement for 50 epochs)
            - Achieve ≥80% mAP@0.5 on validation set
            - Save best model checkpoint by mAP metric
            - Log all metrics to Weights & Biases
          </acceptance_criteria>
          
          <research_benchmarks>
            Reference: IEEE paper on Indian food detection (https://ieeexplore.ieee.org/document/10395249)
            - YOLOv5 achieved 85% mAP with 100 epochs on 12 Indian dishes
            - Best performing: Biryani (92%), Jalebi (88%)
            - Challenging: Samosa (38%), Chole Bhature (79%)
            - Recommendation: Collect 500+ images for low-performing classes
          </research_benchmarks>
        </sub_req>
        
        <sub_req id="FR-5.2">
          <title>Training Monitoring</title>
          <wandb_integration>
            Setup:
              - Platform: Weights & Biases (wandb.ai)
              - Project name: "indian-food-detection"
              - Run name: "yolov8n-mvp-v1"
              - API key: Set via environment variable WANDB_API_KEY
            
            Logged Metrics (per epoch):
              Training:
                - box_loss: Bounding box regression loss
                - cls_loss: Classification loss
                - dfl_loss: Distribution focal loss
                - total_loss: Sum of all losses
              
              Validation:
                - precision: TP / (TP + FP)
                - recall: TP / (TP + FN)
                - mAP@0.5: Mean Average Precision at IoU 0.5
                - mAP@0.5:0.95: mAP averaged over IoU 0.5 to 0.95 (step 0.05)
                - F1-score: Harmonic mean of precision and recall
              
              Per-Class Metrics:
                - Precision, recall, AP for each of 20 food classes
                - Confusion matrix visualization
              
              Learning Rate:
                - Current LR (with OneCycleLR scheduler)
            
            Artifacts:
              - Confusion matrix (end of training)
              - Precision-Recall curves
              - Sample predictions (10 images per epoch)
              - Model checkpoints (best.pt, last.pt)
          </wandb_integration>
          
          <acceptance_criteria>
            - Real-time metric logging during training
            - Generate confusion matrix showing per-class performance
            - Save sample predictions for visual inspection
            - Create PR curves for each class
            - Dashboard accessible via wandb.ai link
          </acceptance_criteria>
        </sub_req>
        
        <sub_req id="FR-5.3">
          <title>Model Export</title>
          <export_formats>
            Primary (MVP):
              - Format: PyTorch (.pt)
              - File: best.pt (best model by mAP)
              - Size: ~6MB
              - Use case: FastAPI inference server
            
            Secondary (Future):
              - Format: ONNX (.onnx)
              - Optimizations: Simplify graph, constant folding
              - Size: ~12MB
              - Use case: Cross-platform deployment (React Native, web)
              
              - Format: TensorFlow Lite (.tflite)
              - Optimizations: INT8 quantization
              - Size: ~3MB
              - Use case: Mobile app (iOS/Android)
          </export_formats>
          
          <acceptance_criteria>
            - Export best.pt after training completes
            - Verify exported model inference matches training results
            - Test inference on 10 sample images
            - Document export commands for future formats
          </acceptance_criteria>
        </sub_req>
      </sub_requirements>
    </requirement>
    
    <requirement id="FR-6" priority="P1_HIGH">
      <category>Nutrition Database</category>
      <description>Store and retrieve nutritional information for Indian foods</description>
      
      <sub_requirements>
        <sub_req id="FR-6.1">
          <title>Custom SQLite Database</title>
          <database_schema>
            CREATE TABLE foods (
              food_id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT UNIQUE NOT NULL,
              category TEXT,
              calories REAL NOT NULL,
              protein REAL NOT NULL,
              carbs REAL NOT NULL,
              fat REAL NOT NULL,
              fiber REAL,
              serving_size TEXT,
              serving_grams REAL NOT NULL,
              cooking_state TEXT CHECK(cooking_state IN ('raw', 'cooked', 'fried', 'steamed')),
              oil_content TEXT CHECK(oil_content IN ('none', 'light', 'medium', 'heavy')),
              source TEXT,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX idx_food_name ON foods(name);
            CREATE INDEX idx_category ON foods(category);
          </database_schema>
          
          <initial_data>
            Sample entries (20 foods minimum):
            
            1. Biryani (chicken)
               - calories: 195, protein: 8.5, carbs: 28, fat: 5.5
               - serving: 100g, cooking_state: cooked, oil: medium
               - source: USDA + Nutritionix average
            
            2. Dhokla
               - calories: 160, protein: 5, carbs: 28, fat: 3
               - serving: 100g, cooking_state: steamed, oil: light
               - source: USDA + manual research
            
            3. Thepla
               - calories: 125, protein: 3, carbs: 18, fat: 5
               - serving: 50g (1 piece), cooking_state: cooked, oil: medium
               - source: Nutritionix
            
            4. Samosa
               - calories: 262, protein: 4.5, carbs: 25, fat: 17
               - serving: 100g (2 pieces), cooking_state: fried, oil: heavy
               - source: USDA
            
            5. Paneer Tikka
               - calories: 180, protein: 12, carbs: 5, fat: 13
               - serving: 100g, cooking_state: cooked, oil: medium
               - source: Nutritionix
            
            ... (15 more entries following same pattern)
            
            Data collection methodology:
            - Primary: USDA FoodData Central
            - Secondary: Nutritionix database
            - Tertiary: Manual calculation from recipes (weighted average)
            - Validation: Cross-check with MyFitnessPal user entries (top 3)
          </initial_data>
          
          <acceptance_criteria>
            - Store 20+ Indian foods with complete nutrition data
            - Include cooking_state and oil_content metadata
            - Support fuzzy name matching (e.g., "biryani" matches "chicken biryani")
            - Query response time under 10ms for local lookups
          </acceptance_criteria>
        </sub_req>
        
        <sub_req id="FR-6.2">
          <title>USDA API Integration</title>
          <api_spec>
            Endpoint: https://api.nal.usda.gov/fdc/v1/foods/search
            Authentication: API key (free tier)
            Rate Limits: 1000 requests/hour, 3600 requests/day
            
            Request Example:
            GET /fdc/v1/foods/search?api_key=YOUR_KEY&query=biryani&pageSize=5
            
            Response Parsing:
            {
              "foods": [
                {
                  "description": "Chicken Biryani",
                  "foodNutrients": [
                    {"nutrientName": "Energy", "value": 195, "unitName": "kcal"},
                    {"nutrientName": "Protein", "value": 8.5, "unitName": "g"},
                    {"nutrientName": "Carbohydrate, by difference", "value": 28, "unitName": "g"},
                    {"nutrientName": "Total lipid (fat)", "value": 5.5, "unitName": "g"}
                  ]
                }
              ]
            }
            
            Error Handling:
            - 429 Too Many Requests → Cache result, retry after 1 minute
            - 404 Not Found → Fall back to default values
            - Network timeout → Return None, log error
          </api_spec>
          
          <acceptance_criteria>
            - Query USDA when food not in custom database
            - Parse nutrition values from JSON response
            - Cache results locally to reduce API calls
            - Return top match based on name similarity (fuzzy matching)
            - Handle rate limiting gracefully
          </acceptance_criteria>
        </sub_req>
        
        <sub_req id="FR-6.3">
          <title>Nutrition Lookup Hierarchy</title>
          <lookup_flow>
            def get_nutrition_data(food_name: str) -> dict:
              # Step 1: Check custom SQLite database
              result = query_custom_db(food_name)
              if result and confidence > 0.9:
                return result
              
              # Step 2: Query USDA API
              usda_result = query_usda_api(food_name)
              if usda_result and confidence > 0.7:
                # Cache to custom DB for future lookups
                cache_to_db(usda_result)
                return usda_result
              
              # Step 3: Return default values (last resort)
              return {
                'name': food_name,
                'calories': 250,  # Average Indian meal
                'protein': 10,
                'carbs': 35,
                'fat': 8,
                'serving_grams': 100,
                'source': 'default_estimate',
                'confidence': 0.3
              }
          </lookup_flow>
          
          <acceptance_criteria>
            - Try custom DB first (fastest, most accurate for Indian foods)
            - Fall back to USDA for unknown foods
            - Use defaults only when all sources fail
            - Log data source for each lookup (debugging)
            - Return confidence score with each result
          </acceptance_criteria>
        </sub_req>
      </sub_requirements>
    </requirement>
    
  </functional_requirements>

  <!-- ============================================ -->
  <!-- NON-FUNCTIONAL REQUIREMENTS -->
  <!-- ============================================ -->
  
  <non_functional_requirements>
    
    <requirement id="NFR-1" category="PERFORMANCE">
      <title>Response Time</title>
      <specifications>
        - Model inference: Under 2 seconds on GPU (NVIDIA T4 or better)
        - Model inference: Under 5 seconds on CPU (Intel i5 or equivalent)
        - API end-to-end: Under 3 seconds (including network, preprocessing, inference, postprocessing)
        - Database queries: Under 10ms for local SQLite lookups
        - USDA API calls: Under 500ms (external dependency)
      </specifications>
      <measurement>
        - Tool: Python time.time() or time.perf_counter()
        - Logging: W&B custom metrics, FastAPI middleware
        - Target: 95th percentile under specified thresholds
      </measurement>
    </requirement>
    
    <requirement id="NFR-2" category="ACCURACY">
      <title>Model Performance</title>
      <specifications>
        Object Detection:
          - mAP@0.5: ≥80% on validation set
          - mAP@0.5:0.95: ≥60% on validation set
          - Precision: ≥85% (minimize false positives)
          - Recall: ≥75% (acceptable false negative rate)
        
        Portion Estimation:
          - Error margin: ±30% acceptable for MVP
          - Measurement: Compare estimated grams vs. actual weight (scale)
          - Sample size: Test on 50 real food photos with known weights
        
        Macro Calculation:
          - Total error: ±25% acceptable (compound of detection + portion + nutrition)
          - Breakdown: Detection (5%) + Portion (20%) + Nutrition DB (5%)
      </specifications>
      <validation_methodology>
        - Gold standard: 50 test meals weighed on food scale
        - Manual macro calculation from ingredients
        - Compare app output vs. ground truth
        - Calculate Mean Absolute Percentage Error (MAPE)
      </validation_methodology>
    </requirement>
    
    <requirement id="NFR-3" category="RELIABILITY">
      <title>System Availability</title>
      <specifications>
        - Uptime: 95%+ (best effort for MVP, no SLA)
        - Error rate: Under 5% of requests should fail
        - Graceful degradation: Return partial results if possible
        - Error messages: Informative, actionable (e.g., "Image too blurry, retake photo")
      </specifications>
      <error_handling>
        - Input validation: Reject invalid image formats/sizes before processing
        - Model errors: Catch exceptions, return 500 with details
        - Database errors: Fall back to USDA API if SQLite fails
        - Network errors: Retry USDA API once with exponential backoff
      </error_handling>
    </requirement>
    
    <requirement id="NFR-4" category="SCALABILITY">
      <title>Concurrent Users (Future)</title>
      <specifications>
        MVP (Week 1):
          - Concurrent requests: 1-5 users (development testing)
          - Deployment: Local machine (laptop/desktop)
        
        V2 (Week 2-4):
          - Concurrent requests: 10-20 users
          - Deployment: Cloud instance (Google Cloud Run, AWS Lambda)
          - Auto-scaling: Based on request volume
        
        Production (Month 2+):
          - Concurrent requests: 100+ users
          - Load balancing: Multiple API instances
          - Caching: Redis for nutrition lookups
          - CDN: Cloudflare for static assets
      </specifications>
    </requirement>
    
    <requirement id="NFR-5" category="MAINTAINABILITY">
      <title>Code Quality & Documentation</title>
      <specifications>
        Code Standards:
          - Python: PEP 8 style guide, type hints
          - Linting: Ruff or Pylint
          - Formatting: Black (line length 100)
          - Docstrings: Google style for all functions
        
        Documentation:
          - README.md: Project overview, setup instructions
          - API.md: Endpoint documentation with examples
          - TRAINING.md: Model training guide, hyperparameters
          - METRICS.md: Performance benchmarks, accuracy report
        
        Version Control:
          - Git repository with .gitignore (exclude models, data)
          - Commit messages: Conventional Commits format
          - Branching: main (stable), dev (active development)
        
        Testing:
          - Unit tests: Core utility functions (pytest)
          - Integration tests: API endpoints (FastAPI TestClient)
          - Model tests: Inference on sample images
          - Coverage: 60%+ for MVP, 80%+ for production
      </specifications>
    </requirement>
    
    <requirement id="NFR-6" category="SECURITY">
      <title>Data Privacy & API Security</title>
      <specifications>
        MVP:
          - No user authentication (stateless API)
          - No data persistence (images discarded after processing)
          - CORS: Allow all origins (*)
        
        V2:
          - API key authentication for production
          - Rate limiting: 100 requests/hour per IP
          - Input sanitization: Validate file types, sizes
          - HTTPS only: SSL/TLS encryption
        
        Data Handling:
          - Uploaded images: Stored in memory only, never saved to disk
          - Logs: Strip any PII (user data, IP addresses)
          - Nutrition data: Public domain (USDA) or open source
      </specifications>
    </requirement>
    
    <requirement id="NFR-7" category="USABILITY">
      <title>Developer Experience</title>
      <specifications>
        Setup Time:
          - Clone repository to working API: Under 30 minutes
          - Prerequisites: Python 3.10+, pip, (optional) GPU with CUDA
        
        Ease of Use:
          - Single command training: `python scripts/train_model.py`
          - Single command server start: `uvicorn backend.main:app --reload`
          - Environment variables: .env file for API keys, paths
        
        Debugging:
          - Verbose logging: Configurable via LOG_LEVEL env var
          - Error stack traces: Full details in development mode
          - Sample test images: Included in repository
      </specifications>
    </requirement>
    
  </non_functional_requirements>

  <!-- ============================================ -->
  <!-- OPEN SOURCE MODELS & LIBRARIES -->
  <!-- ============================================ -->
  
  <open_source_resources>
    
    <computer_vision_models>
      <model id="YOLO-1">
        <name>YOLOv8 Nano</name>
        <source>Ultralytics (https://github.com/ultralytics/ultralytics)</source>
        <license>AGPL-3.0 (open source)</license>
        <pretrained_weights>yolov8n.pt (COCO dataset, 80 classes)</pretrained_weights>
        <parameters>3.2M (smallest YOLO variant)</parameters>
        <speed>200+ FPS on GPU, 37 FPS on CPU</speed>
        <accuracy>37.3 mAP@0.5 on COCO (baseline before fine-tuning)</accuracy>
        <use_case>Primary object detection and classification model</use_case>
        <why_chosen>
          - State-of-the-art accuracy-speed tradeoff
          - Mobile-friendly (small model size)
          - Active development and community support
          - Easy to fine-tune with custom datasets
          - Research shows 85% accuracy on Indian food (IEEE paper)
        </why_chosen>
      </model>
      
      <model id="DEPTH-1">
        <name>MiDaS v3.1 (DPT-Small)</name>
        <source>Intel ISL (https://github.com/isl-org/MiDaS)</source>
        <license>MIT (open source)</license>
        <pretrained_weights>Available via torch.hub.load</pretrained_weights>
        <use_case>Depth estimation for portion size calculation (v2 feature)</use_case>
        <status>Deferred to v2 - MVP uses bbox area heuristic</status>
        <why_deferred>
          - Adds complexity to MVP
          - Requires additional inference time (200-500ms)
          - Bbox area provides acceptable ±30% accuracy for MVP
        </why_deferred>
      </model>
      
      <model id="CLASSIFIER-1">
        <name>EfficientNetB0</name>
        <source>TensorFlow/Keras (https://keras.io/api/applications/)</source>
        <license>Apache 2.0 (open source)</license>
        <pretrained_weights>ImageNet (1000 classes)</pretrained_weights>
        <use_case>Fine-grained food classification (v2 feature)</use_case>
        <status>Deferred to v2 - YOLO classification sufficient for MVP</status>
        <why_deferred>
          - YOLOv8 already provides classification
          - Two-model pipeline adds latency
          - Useful for ambiguous cases (samosa vs. kachori)
        </why_deferred>
      </model>
    </computer_vision_models>
    
    <python_libraries>
      <library id="LIB-1">
        <name>Ultralytics</name>
        <version>8.0.0+</version>
        <purpose>YOLOv8 training and inference</purpose>
        <install>pip install ultralytics</install>
        <license>AGPL-3.0</license>
      </library>
      
      <library id="LIB-2">
        <name>FastAPI</name>
        <version>0.100.0+</version>
        <purpose>Web framework for API backend</purpose>
        <install>pip install fastapi[all]</install>
        <license>MIT</license>
      </library>
      
      <library id="LIB-3">
        <name>Uvicorn</name>
        <version>0.22.0+</version>
        <purpose>ASGI server for FastAPI</purpose>
        <install>pip install uvicorn[standard]</install>
        <license>BSD-3-Clause</license>
      </library>
      
      <library id="LIB-4">
        <name>Pillow (PIL)</name>
        <version>10.0.0+</version>
        <purpose>Image processing and manipulation</purpose>
        <install>pip install Pillow</install>
        <license>HPND (PIL License)</license>
      </library>
      
      <library id="LIB-5">
        <name>OpenCV (cv2)</name>
        <version>4.8.0+</version>
        <purpose>Advanced image processing</purpose>
        <install>pip install opencv-python</install>
        <license>Apache 2.0</license>
      </library>
      
      <library id="LIB-6">
        <name>NumPy</name>
        <version>1.24.0+</version>
        <purpose>Numerical computing, array operations</purpose>
        <install>pip install numpy</install>
        <license>BSD-3-Clause</license>
      </library>
      
      <library id="LIB-7">
        <name>PyTorch</name>
        <version>2.0.0+</version>
        <purpose>Deep learning framework (YOLOv8 backend)</purpose>
        <install>pip install torch torchvision</install>
        <license>BSD-3-Clause</license>
      </library>
      
      <library id="LIB-8">
        <name>Weights & Biases (wandb)</name>
        <version>0.15.0+</version>
        <purpose>Experiment tracking and visualization</purpose>
        <install>pip install wandb</install>
        <license>MIT</license>
        <account>Free tier sufficient (100GB storage, unlimited runs)</account>
      </library>
      
      <library id="LIB-9">
        <name>Roboflow</name>
        <version>1.0.0+</version>
        <purpose>Dataset download and management</purpose>
        <install>pip install roboflow</install>
        <license>Apache 2.0</license>
      </library>
      
      <library id="LIB-10">
        <name>Kaggle</name>
        <version>1.5.0+</version>
        <purpose>Kaggle dataset download</purpose>
        <install>pip install kaggle</install>
        <license>Apache 2.0</license>
      </library>
      
      <library id="LIB-11">
        <name>Requests</name>
        <version>2.31.0+</version>
        <purpose>HTTP requests for USDA API</purpose>
        <install>pip install requests</install>
        <license>Apache 2.0</license>
      </library>
      
      <library id="LIB-12">
        <name>SQLite3</name>
        <version>Built-in (Python standard library)</version>
        <purpose>Local nutrition database</purpose>
        <install>No installation needed</install>
        <license>Public Domain</license>
      </library>
      
      <library id="LIB-13">
        <name>python-dotenv</name>
        <version>1.0.0+</version>
        <purpose>Environment variable management</purpose>
        <install>pip install python-dotenv</install>
        <license>BSD-3-Clause</license>
      </library>
      
      <library id="LIB-14">
        <name>FuzzyWuzzy</name>
        <version>0.18.0+</version>
        <purpose>Fuzzy string matching for food names</purpose>
        <install>pip install fuzzywuzzy python-Levenshtein</install>
        <license>GPL-2.0</license>
      </library>
    </python_libraries>
    
  </open_source_resources>

  <!-- ============================================ -->
  <!-- FREE TOOLING & SERVICES -->
  <!-- ============================================ -->
  
  <free_tooling>
    
    <development_tools>
      <tool id="TOOL-1">
        <name>Google Colab</name>
        <purpose>GPU-accelerated model training</purpose>
        <specs>
          - Free tier: NVIDIA T4 GPU (16GB VRAM), 12GB RAM
          - Session limit: 12 hours continuous
          - Storage: 15GB Google Drive integration
          - Ideal for: Training YOLOv8 (100 epochs ~4-6 hours on T4)
        </specs>
        <url>https://colab.research.google.com/</url>
        <cost>Free (with limitations)</cost>
      </tool>
      
      <tool id="TOOL-2">
        <name>Weights & Biases</name>
        <purpose>Experiment tracking, metric visualization, model versioning</purpose>
        <specs>
          - Free tier: 100GB storage, unlimited runs
          - Features: Real-time dashboards, hyperparameter sweeps, artifact management
          - Integrations: PyTorch, Ultralytics, Keras
        </specs>
        <url>https://wandb.ai/</url>
        <cost>Free for individuals</cost>
      </tool>
      
      <tool id="TOOL-3">
        <name>Roboflow</name>
        <purpose>Dataset hosting, annotation, augmentation, export</purpose>
        <specs>
          - Free tier: 10,000 source images, 3 projects
          - Features: Auto-annotation, format conversion (YOLO, COCO, etc.)
          - API: Python SDK for programmatic dataset download
        </specs>
        <url>https://roboflow.com/</url>
        <cost>Free tier available</cost>
      </tool>
      
      <tool id="TOOL-4">
        <name>Kaggle</name>
        <purpose>Dataset hosting and GPU notebooks</purpose>
        <specs>
          - Free tier: Unlimited dataset downloads, 30 hours/week GPU (P100)
          - API: Command-line tool for dataset downloads
          - Community: Notebooks, competitions, discussions
        </specs>
        <url>https://www.kaggle.com/</url>
        <cost>Free</cost>
      </tool>
      
      <tool id="TOOL-5">
        <name>USDA FoodData Central API</name>
        <purpose>Comprehensive nutrition database</purpose>
        <specs>
          - Free tier: 1000 requests/hour, 3600/day
          - Data: 400,000+ foods with detailed nutrition info
          - Coverage: Global foods, branded products, USDA standards
        </specs>
        <url>https://fdc.nal.usda.gov/api-guide.html</url>
        <cost>Free (API key required)</cost>
      </tool>
      
      <tool id="TOOL-6">
        <name>Label Studio</name>
        <purpose>Open-source data annotation (alternative to Roboflow)</purpose>
        <specs>
          - Features: Bounding boxes, segmentation, classification
          - Self-hosted: Run locally, no cloud dependency
          - Export: YOLO, COCO, Pascal VOC formats
        </specs>
        <url>https://labelstud.io/</url>
        <cost>Free (open source)</cost>
      </tool>
      
      <tool id="TOOL-7">
        <name>VS Code + Cursor</name>
        <purpose>Code editor with AI assistance</purpose>
        <specs>
          - Extensions: Python, Jupyter, GitLens, Docker
          - AI: GitHub Copilot, Cursor AI (context-aware code generation)
          - Debugging: Integrated Python debugger, breakpoints
        </specs>
        <url>https://code.visualstudio.com/, https://cursor.sh/</url>
        <cost>Free (Copilot requires subscription)</cost>
      </tool>
    </development_tools>
    
    <deployment_options>
      <option id="DEPLOY-1">
        <name>Local Development Server</name>
        <specs>
          - Hardware: Laptop/desktop with 8GB+ RAM
          - GPU: Optional (NVIDIA with CUDA for faster inference)
          - Cost: $0 (using existing hardware)
        </specs>
        <use_case>MVP testing, Week 1-2</use_case>
      </option>
      
      <option id="DEPLOY-2">
        <name>Google Cloud Run</name>
        <specs>
          - Free tier: 2M requests/month, 360,000 GB-seconds
          - Auto-scaling: 0 to N instances based on load
          - Cold start: ~5 seconds (acceptable for MVP)
        </specs>
        <use_case>V2 deployment (Week 3+)</use_case>
        <cost>Free tier likely sufficient for MVP</cost>
      </option>
      
      <option id="DEPLOY-3">
        <name>AWS Lambda + API Gateway</name>
        <specs>
          - Free tier: 1M requests/month, 400,000 GB-seconds
          - Constraints: 10GB memory max, 15-minute timeout
          - Model loading: Use Lambda layers or EFS
        </specs>
        <use_case>Alternative to Cloud Run</use_case>
        <cost>Free tier available</cost>
      </option>
    </deployment_options>
    
  </free_tooling>

  <!-- ============================================ -->
  <!-- DATA COLLECTION STRATEGY -->
  <!-- ============================================ -->
  
  <data_collection>
    
    <phase_1_existing_datasets>
      <workflow>
        Step 1: Setup API Credentials
          - Roboflow: Create account, get API key from roboflow.com/settings
          - Kaggle: Download kaggle.json from kaggle.com/settings, place in ~/.kaggle/
        
        Step 2: Download Datasets
          Script: scripts/download_datasets.py
          
          def download_all_datasets():
            # Roboflow dataset 1
            rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
            project1 = rf.workspace("datacluster-labs-agryi").project("indian-food-image")
            dataset1 = project1.version(1).download("yolov8")
            
            # Roboflow dataset 2
            project2 = rf.workspace("smart-india-hackathon").project("indian-food-yolov5")
            dataset2 = project2.version(1).download("yolov8")
            
            # Kaggle datasets
            kaggle.api.dataset_download_files('dataclusterlabs/indian-food-image-dataset', 
                                              path='./data/raw/kaggle_indian_food', unzip=True)
            kaggle.api.dataset_download_files('sukhmandeepsinghbrar/indian-food-dataset',
                                              path='./data/raw/kaggle_metadata', unzip=True)
          
        Step 3: Verify Downloads
          - Check directory structure matches YOLO format
          - Count images and labels (should match)
          - Random sample inspection (10 images)
          - Log dataset statistics (class distribution, image sizes)
        
        Step 4: Consolidate
          - Merge datasets into ./data/processed/
          - Resolve class name conflicts
          - Create unified data.yaml with 20 classes
          - Split: 70% train, 20% val, 10% test (if not pre-split)
      </workflow>
      
      <expected_output>
        Total images: 9,000-10,000 (combined from all sources)
        Classes: 20 Indian dishes
        Format: YOLO v8 (class_id, center_x, center_y, width, height)
        Structure:
          data/processed/
          ├── train/ (7000 images)
          ├── valid/ (2000 images)
          ├── test/ (1000 images)
          └── data.yaml
      </expected_output>
    </phase_1_existing_datasets>
    
    <phase_2_custom_augmentation>
      <status>Deferred to V2+</status>
      <rationale>Existing datasets provide sufficient data for MVP (9K+ images)</rationale>
      <future_collection>
        Week 3+:
        - Personal photos: 50-100 images from Toronto Indian restaurants
        - Focus on underrepresented dishes (based on training metrics)
        - Google Images scraping: Augment classes with low accuracy
        - Instagram hashtag scraping: #gujaratifood, #indianfoodie
      </future_collection>
    </phase_2_custom_augmentation>
    
    <annotation_tools>
      <primary>
        <name>Roboflow (Cloud-based)</name>
        <use_case>If additional annotation needed</use_case>
        <features>
          - Auto-annotation with base models
          - Collaborative labeling
          - One-click YOLO export
        </features>
      </primary>
      
      <alternative>
        <name>Label Studio (Self-hosted)</name>
        <use_case>If privacy concerns or offline annotation needed</use_case>
        <setup>
          pip install label-studio
          label-studio start ./indian_food_project
        </setup>
        <features>
          - Bounding box annotation
          - Keyboard shortcuts for speed
          - Export to YOLO format
        </features>
      </alternative>
    </annotation_tools>
    
  </data_collection>

  <!-- ============================================ -->
  <!-- MVP DEVELOPMENT PHASES -->
  <!-- ============================================ -->
  
  <development_phases>
    
    <phase id="PHASE-0" duration="1 day">
      <name>Project Setup & Data Acquisition</name>
      <tasks>
        1. Repository initialization
           - Create GitHub repo: indian-food-macro-tracker
           - Initialize with README.md, .gitignore, LICENSE
           - Set up directory structure (data/, models/, backend/, scripts/)
        
        2. Environment setup
           - Create Python 3.10+ virtual environment
           - Install dependencies: pip install -r requirements.txt
           - Configure environment variables (.env file)
        
        3. Dataset download
           - Run scripts/download_datasets.py
           - Verify YOLO format with scripts/prepare_yolo_format.py
           - Create data.yaml configuration
        
        4. Tool setup
           - Weights & Biases: Create account, get API key
           - USDA API: Register, get API key
           - Google Colab: Test GPU access
      </tasks>
      
      <deliverables>
        - ✓ Working development environment
        - ✓ 9K+ images in YOLO format
        - ✓ data.yaml with 20 classes
        - ✓ W&B and USDA accounts configured
      </deliverables>
      
      <success_criteria>
        - Can load sample image with PIL/cv2
        - Can parse YOLO label file
        - W&B test run logs successfully
      </success_criteria>
    </phase>
    
    <phase id="PHASE-1" duration="2-3 days">
      <name>Model Training & Validation</name>
      <tasks>
        1. Google Colab notebook setup
           - Mount Google Drive
           - Upload data.yaml and datasets (or download via Roboflow API)
           - Install ultralytics, wandb
        
        2. YOLOv8 fine-tuning
           - Load yolov8n.pt pretrained weights
           - Configure hyperparameters (epochs=100, batch=16, etc.)
           - Start training with W&B logging
           - Monitor loss curves, mAP in real-time
        
        3. Model evaluation
           - Run validation on test set
           - Generate confusion matrix
           - Analyze per-class performance (identify weak classes)
           - Visualize sample predictions
        
        4. Model export
           - Save best.pt checkpoint
           - Download to local machine
           - Test inference on sample images
      </tasks>
      
      <deliverables>
        - ✓ Trained YOLOv8 model (best.pt)
        - ✓ W&B dashboard with training metrics
        - ✓ Confusion matrix and PR curves
        - ✓ Model achieves 80%+ mAP@0.5
      </deliverables>
      
      <success_criteria>
        - Training completes without errors
        - mAP@0.5 ≥ 80% on validation set
        - Inference runs on CPU in under 5 seconds
        - Sample predictions look reasonable
      </success_criteria>
    </phase>
    
    <phase id="PHASE-2" duration="2-3 days">
      <name>Backend API Development</name>
      <tasks>
        1. Nutrition database setup
           - Create gujarati_foods.db SQLite database
           - Populate with 20 Indian foods (manual research)
           - Implement query functions with fuzzy matching
        
        2. USDA API integration
           - Write usda_client.py with API wrapper
           - Test queries for common foods
           - Implement caching to reduce API calls
        
        3. FastAPI server implementation
           - Create main.py with /analyze-food endpoint
           - Implement image upload handling
           - Integrate YOLOv8 inference
           - Connect nutrition lookup (SQLite → USDA → default)
           - Calculate total macros
        
        4. Portion estimation (simple)
           - Implement bbox area → grams conversion
           - Use calibration constants (pixel-to-cm ratio, density)
           - Clamp to reasonable ranges (50g-500g)
        
        5. Testing
           - Test with 20 sample food images
           - Verify JSON response format
           - Check error handling (invalid images, no detections)
      </tasks>
      
      <deliverables>
        - ✓ FastAPI server running on localhost:8000
        - ✓ /analyze-food endpoint functional
        - ✓ SQLite nutrition database with 20 foods
        - ✓ USDA API integration working
      </deliverables>
      
      <success_criteria>
        - curl upload returns valid JSON response
        - API processes image in under 3 seconds
        - Nutrition lookups succeed for all 20 dishes
        - Error responses are informative
      </success_criteria>
    </phase>
    
    <phase id="PHASE-3" duration="1-2 days">
      <name>Testing & Documentation</name>
      <tasks>
        1. Accuracy testing
           - Collect 50 real food photos (personal or online)
           - Run through API, collect predictions
           - Manually verify detections and macro estimates
           - Calculate error rates (detection, portion, total)
        
        2. Performance testing
           - Measure API latency (end-to-end)
           - Test on CPU vs. GPU
           - Identify bottlenecks (model loading, inference, nutrition lookup)
        
        3. Documentation
           - Write README.md (setup, usage, troubleshooting)
           - Document API endpoints (API.md)
           - Training guide (TRAINING.md)
           - Metrics report (METRICS.md with accuracy results)
        
        4. Code cleanup
           - Add docstrings to all functions
           - Format with Black
           - Lint with Ruff/Pylint
           - Commit to GitHub
      </tasks>
      
      <deliverables>
        - ✓ Accuracy report on 50 test images
        - ✓ Complete project documentation
        - ✓ Clean, well-documented code
        - ✓ GitHub repository ready to share
      </deliverables>
      
      <success_criteria>
        - Detection accuracy ≥ 75% on real-world photos
        - Total macro error ≤ 30%
        - Documentation covers all setup steps
        - New developer can run project in under 30 minutes
      </success_criteria>
    </phase>
    
  </development_phases>

  <!-- ============================================ -->
  <!-- DEFERRED FEATURES (V2+) -->
  <!-- ============================================ -->
  
  <deferred_features>
    <feature id="DEFER-1" priority="V2">
      <name>Depth-based Portion Estimation</name>
      <description>Use MiDaS depth estimation for more accurate volume calculation</description>
      <rationale>MVP bbox area heuristic acceptable for ±30% accuracy</rationale>
      <implementation_plan>
        Week 3-4:
        - Integrate MiDaS DPT-Small model
        - Calculate volume from depth map + bbox
        - Calibrate with 100 weighed food samples
        - Target: ±15% portion estimation accuracy
      </implementation_plan>
    </feature>
    
    <feature id="DEFER-2" priority="V2">
      <name>Mobile App (React Native/Expo)</name>
      <description>Native mobile app with camera integration</description>
      <rationale>MVP focuses on backend; web upload form sufficient for testing</rationale>
      <implementation_plan>
        Week 4-6:
        - Set up Expo project
        - Implement camera screen (expo-camera)
        - Results screen with macro visualization
        - Connect to FastAPI backend
        - Deploy to TestFlight/Play Store beta
      </implementation_plan>
    </feature>
    
    <feature id="DEFER-3" priority="V2">
      <name>User Feedback Loop</name>
      <description>Allow users to correct portion estimates, improve model over time</description>
      <rationale>Requires database, user accounts, analytics infrastructure</rationale>
      <implementation_plan>
        Month 2:
        - Add user authentication (Firebase/Auth0)
        - Correction UI (slider for portion adjustment)
        - Store corrections in PostgreSQL
        - Retrain model monthly with corrected data
      </implementation_plan>
    </feature>
    
    <feature id="DEFER-4" priority="V3">
      <name>Cooking State Classification</name>
      <description>Detect raw vs. cooked vs. fried foods, adjust macros accordingly</description>
      <rationale>Complex; most food photos are of cooked meals</rationale>
      <implementation_plan>
        Month 3+:
        - Collect dataset with cooking state labels
        - Add classification head to YOLOv8
        - Apply cooking multipliers (rice 2.5x, chicken 0.75x)
        - Test accuracy improvement
      </implementation_plan>
    </feature>
    
    <feature id="DEFER-5" priority="V3">
      <name>Multi-Model Ensemble (YOLO + EfficientNet)</name>
      <description>Use EfficientNetB0 for fine-grained classification of ambiguous detections</description>
      <rationale>YOLO sufficient for MVP; ensemble adds latency</rationale>
      <implementation_plan>
        Month 4+:
        - Train EfficientNetB0 on 50+ classes
        - Implement two-stage pipeline (YOLO detect → EfficientNet classify)
        - Benchmark accuracy vs. latency tradeoff
        - Use only for confidence < 0.7 detections
      </implementation_plan>
    </feature>
  </deferred_features>

  <!-- ============================================ -->
  <!-- SUCCESS METRICS & KPIs -->
  <!-- ============================================ -->
  
  <success_metrics>
    
    <mvp_launch_criteria>
      <criterion id="METRIC-1">
        <name>Model Accuracy</name>
        <target>≥80% mAP@0.5 on validation set</target>
        <measurement>YOLOv8 validation metrics from W&B</measurement>
        <status>Required for MVP launch</status>
      </criterion>
      
      <criterion id="METRIC-2">
        <name>API Response Time</name>
        <target>Under 3 seconds end-to-end (95th percentile)</target>
        <measurement>FastAPI middleware timing logs</measurement>
        <status>Required for MVP launch</status>
      </criterion>
      
      <criterion id="METRIC-3">
        <name>Real-World Testing</name>
        <target>Test on 50 real food photos with manual verification</target>
        <measurement>Manual comparison of predictions vs. ground truth</measurement>
        <status>Required for MVP launch</status>
      </criterion>
      
      <criterion id="METRIC-4">
        <name>Documentation Completeness</name>
        <target>New developer can run project in under 30 minutes</target>
        <measurement>Walkthrough with uninvolved third party</measurement>
        <status>Required for MVP launch</status>
      </criterion>
    </mvp_launch_criteria>
    
    <post_launch_kpis>
      <kpi id="KPI-1">
        <name>Detection Success Rate</name>
        <definition>Percentage of images with at least one valid detection (confidence > 0.5)</definition>
        <target>≥85% of submitted images</target>
        <tracking>Log all API requests, count successes vs. failures</tracking>
      </kpi>
      
      <kpi id="KPI-2">
        <name>Macro Estimation Error</name>
        <definition>Mean Absolute Percentage Error (MAPE) on total macros vs. ground truth</definition>
        <target>≤30% for MVP, ≤20% for V2</target>
        <tracking>Weekly manual testing with 10 weighed meals</tracking>
      </kpi>
      
      <kpi id="KPI-3">
        <name>User Satisfaction (Qualitative)</name>
        <definition>Feedback from early testers (friends, family, Reddit r/MacroTracking)</definition>
        <target>70%+ report "useful" or "very useful"</target>
        <tracking>Google Form survey after 1 week of use</tracking>
      </kpi>
      
      <kpi id="KPI-4">
        <name>API Uptime</name>
        <definition>Percentage of time API returns 200 status for health check</definition>
        <target>≥95% uptime</target>
        <tracking>UptimeRobot or similar monitoring service</tracking>
      </kpi>
    </post_launch_kpis>
    
  </success_metrics>

  <!-- ============================================ -->
  <!-- RISK MITIGATION -->
  <!-- ============================================ -->
  
  <risks_and_mitigation>
    
    <risk id="RISK-1" severity="HIGH">
      <description>Model accuracy below 80% on validation set</description>
      <probability>Medium (research shows 85% achievable, but dataset may differ)</probability>
      <impact>MVP launch delayed, need more data or hyperparameter tuning</impact>
      <mitigation>
        - Start training early (Day 2-3) to allow iteration time
        - If mAP < 80% after 100 epochs:
          1. Analyze confusion matrix (identify weak classes)
          2. Collect 200+ more images for low-performing dishes
          3. Increase augmentation (stronger HSV, mosaic, mixup)
          4. Try YOLOv8s (larger model) if compute allows
        - Fallback: Lower threshold to 75% for MVP if user testing shows acceptable results
      </mitigation>
    </risk>
    
    <risk id="RISK-2" severity="MEDIUM">
      <description>API response time exceeds 3 seconds on CPU</description>
      <probability>Medium (YOLOv8n designed for speed, but depends on hardware)</probability>
      <impact>Poor user experience, need optimization or GPU deployment</impact>
      <mitigation>
        - Benchmark inference time during Phase 2
        - If too slow:
          1. Use ONNX export with optimizations
          2. Reduce image size (640 → 480 or 320)
          3. Deploy to Google Cloud Run with GPU (free tier)
          4. Implement caching for repeated images (hash-based lookup)
        - Fallback: Set expectation in docs ("Designed for GPU, CPU may be slow")
      </mitigation>
    </risk>
    
    <risk id="RISK-3" severity="LOW">
      <description>USDA API rate limiting (1000 req/hour) insufficient</description>
      <probability>Low (MVP has few users)</probability>
      <impact>Nutrition lookups fail during high usage</impact>
      <mitigation>
        - Implement aggressive caching (cache all USDA responses in SQLite)
        - Monitor API usage with logging
        - If rate limit hit:
          1. Use cached data exclusively for 1 hour
          2. Expand custom nutrition database to 50+ foods
          3. Consider paid USDA plan ($50/month for 10K req/hour)
      </mitigation>
    </risk>
    
    <risk id="RISK-4" severity="LOW">
      <description>Dataset download fails (Roboflow/Kaggle API issues)</description>
      <probability>Low (APIs are stable)</probability>
      <impact>Phase 0 delayed, can't start training</impact>
      <mitigation>
        - Test dataset download on Day 1
        - If API fails:
          1. Manual download via browser
          2. Use alternate datasets (Hugging Face, paperswithcode.com)
          3. Contact dataset authors directly
        - Keep copy of datasets in Google Drive as backup
      </mitigation>
    </risk>
    
  </risks_and_mitigation>

  <!-- ============================================ -->
  <!-- AGENT INSTRUCTIONS -->
  <!-- ============================================ -->
  
  <agent_instructions>
    <role>AI Software Development Agent (Google Gemini Code Assist)</role>
    <persona>
      You are an expert full-stack ML engineer with deep knowledge of:
      - Computer vision (YOLO, object detection, image classification)
      - Python (FastAPI, PyTorch, Ultralytics)
      - ML Ops (training workflows, model deployment, monitoring)
      - Mobile development (React Native, Expo)
      
      Your communication style:
      - Concise, actionable code with detailed comments
      - Proactive error handling and edge case coverage
      - Production-ready patterns (logging, validation, testing)
      - Educational explanations for key decisions
    </persona>
    
    <task_prioritization>
      1. **Correctness**: Code must work as specified in requirements
      2. **Completeness**: No TODOs or placeholders in production code
      3. **Clarity**: Readable, well-documented, easy to debug
      4. **Performance**: Optimize for speed where specified (e.g., API latency)
      5. **Maintainability**: Modular design, separation of concerns
    </task_prioritization>
    
    <code_generation_rules>
      - Always include type hints for function signatures
      - Add docstrings (Google style) for all public functions
      - Use descriptive variable names (no single letters except loop indices)
      - Include error handling (try/except with specific exceptions)
      - Log important events (model loading, API calls, errors)
      - Follow PEP 8 style guide (line length 100, Black formatting)
      - Avoid premature optimization (MVP first, optimize later if needed)
    </code_generation_rules>
    
    <workflow>
      When given a task:
      
      1. **Understand Context**
         - Review relevant sections of this spec
         - Identify functional and non-functional requirements
         - Note any constraints (timeline, accuracy, budget)
      
      2. **Plan Approach**
         - Break task into subtasks
         - Identify dependencies (data, models, APIs)
         - Choose appropriate libraries and patterns
      
      3. **Generate Code**
         - Write complete, working code (no pseudocode)
         - Include imports, error handling, logging
         - Add inline comments for complex logic
         - Follow code_generation_rules above
      
      4. **Validate Output**
         - Check against acceptance criteria
         - Verify edge cases handled
         - Ensure matches specified format/API
         - Add unit tests if applicable
      
      5. **Document**
         - Explain key decisions in comments
         - Note any assumptions or limitations
         - Suggest next steps or improvements
    </workflow>
    
    <example_prompts>
      <!-- Use these XML prompt templates for each phase -->
      
      <prompt_template id="TEMPLATE-1">
        <phase>Phase 0: Dataset Download</phase>
        <content>
          <project>
            <name>Indian Food Macro Tracker</name>
            <current_phase>Phase 0: Data Acquisition</current_phase>
            
            <task>
              <description>Generate complete Python script to download Indian food datasets from Roboflow and Kaggle</description>
              <input>
                - Roboflow API key (from environment variable)
                - Kaggle credentials (from ~/.kaggle/kaggle.json)
              </input>
              <output>
                - Script: scripts/download_datasets.py
                - Downloads 4 datasets to ./data/raw/
                - Logs progress and completion status
              </output>
              <requirements>
                - Download Roboflow dataset 1: datacluster-labs-agryi/indian-food-image
                - Download Roboflow dataset 2: smart-india-hackathon/indian-food-yolov5
                - Download Kaggle dataset 1: dataclusterlabs/indian-food-image-dataset
                - Download Kaggle dataset 2: sukhmandeepsinghbrar/indian-food-dataset
                - Verify downloads (check file counts, log statistics)
                - Handle errors gracefully (network issues, invalid API keys)
              </requirements>
            </task>
          </project>
          
          Generate production-ready Python code with error handling, logging, and docstrings.
        </content>
      </prompt_template>
      
      <prompt_template id="TEMPLATE-2">
        <phase>Phase 1: Model Training</phase>
        <content>
          <project>
            <name>Indian Food Macro Tracker</name>
            <current_phase>Phase 1: YOLOv8 Training</current_phase>
            
            <context>
              <dataset>
                - Location: ./data/processed/data.yaml
                - Classes: 20 Indian dishes
                - Images: 9000+ (70/20/10 split)
                - Format: YOLO v8
              </dataset>
              
              <model>
                - Architecture: YOLOv8n (yolov8n.pt pretrained)
                - Target: 80%+ mAP@0.5
                - Training: 100 epochs, batch 16, img 640
              </model>
            </context>
            
            <task>
              <description>Generate complete training script for YOLOv8 fine-tuning on Indian food dataset</description>
              <output>Script: scripts/train_model.py</output>
              <requirements>
                - Load yolov8n.pt pretrained weights
                - Configure hyperparameters per spec (epochs=100, augmentation, etc.)
                - Integrate Weights & Biases logging
                - Save best model checkpoint (best.pt)
                - Generate confusion matrix and sample predictions
                - Export to ONNX format (optional)
              </requirements>
            </task>
          </project>
          
          Generate production-ready training script with W&B integration, error handling, and progress logging.
        </content>
      </prompt_template>
      
      <prompt_template id="TEMPLATE-3">
        <phase>Phase 2: Backend API</phase>
        <content>
          <project>
            <name>Indian Food Macro Tracker</name>
            <current_phase>Phase 2: FastAPI Backend</current_phase>
            
            <context>
              <model>
                - Path: ./models/best.pt
                - Type: YOLOv8n trained on Indian food
                - Output: {food_name, confidence, bbox}
              </model>
              
              <nutrition_db>
                - SQLite: ./data/gujarati_foods.db
                - Fallback: USDA API (fdc.nal.usda.gov)
              </nutrition_db>
            </context>
            
            <task>
              <description>Generate complete FastAPI server with food detection and macro estimation</description>
              <output>File: backend/main.py</output>
              <requirements>
                - Endpoint: POST /analyze-food (multipart/form-data, 'image' field)
                - Load YOLOv8 model at startup (global variable)
                - Run inference on uploaded image
                - Estimate portion size (bbox area heuristic)
                - Lookup nutrition (SQLite → USDA → default)
                - Calculate total macros
                - Return JSON: {detected_foods, total_macros, num_items}
                - Health check endpoint: GET /health
                - CORS enabled for web clients
                - Error handling (invalid images, no detections, API failures)
              </requirements>
            </task>
          </project>
          
          Generate production-ready FastAPI code with async/await, error handling, logging, and OpenAPI docs.
        </content>
      </prompt_template>
    </example_prompts>
    
  </agent_instructions>

  <!-- ============================================ -->
  <!-- QUICK REFERENCE -->
  <!-- ============================================ -->
  
  <quick_reference>
    <commands>
      # Setup
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -r requirements.txt
      
      # Data Download
      python scripts/download_datasets.py
      python scripts/prepare_yolo_format.py
      
      # Training (Google Colab recommended)
      python scripts/train_model.py
      
      # Backend
      uvicorn backend.main:app --reload --port 8000
      
      # Testing
      curl -X POST "http://localhost:8000/analyze-food" -F "image=@test_image.jpg"
      
      # W&B Login
      wandb login YOUR_API_KEY
    </commands>
    
    <key_files>
      requirements.txt          # Python dependencies
      .env                      # Environment variables (API keys)
      data/data.yaml           # YOLO dataset configuration
      models/best.pt           # Trained YOLOv8 model
      backend/main.py          # FastAPI server
      backend/nutrition/gujarati_foods.db  # SQLite nutrition database
    </key_files>
    
    <important_urls>
      Roboflow API: https://docs.roboflow.com/python
      USDA FoodData Central: https://fdc.nal.usda.gov/api-guide.html
      Ultralytics Docs: https://docs.ultralytics.com/
      FastAPI Docs: https://fastapi.tiangolo.com/
      Weights & Biases: https://wandb.ai/
    </important_urls>
  </quick_reference>

</project_specification>
