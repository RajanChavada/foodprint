# Indian Food Macro Tracker (Foodprint)

A computer vision application that detects Indian food dishes from images and estimates nutritional macros (calories, protein, carbs, fat).

## Features
- **Food Detection**: Identifies 20+ Indian dishes using YOLOv8.
- **Macro Estimation**: Calculates nutrition based on portion size and a custom database.
- **API**: FastAPI backend for easy integration.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    - Copy `.env` to `.env.local` (if you want to keep it out of git) or just edit `.env`.
    - Add your API keys for Roboflow, Weights & Biases, and USDA.

3.  **Data Collection**:
    ```bash
    python scripts/download_data.py
    python scripts/prepare_data.py
    ```

4.  **Training**:
    ```bash
    python scripts/train_model.py
    ```

5.  **Run API**:
    ```bash
    uvicorn backend.main:app --reload
    ```
