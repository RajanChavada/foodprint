from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()

def train_model():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data='data/processed/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=50,
        project='models',
        name='foodprint_v1',
        exist_ok=True,
        device='cpu', # Force CPU for local training if no GPU
        # device='0' # Use this if you have a GPU
        mode='disabled' # Disable W&B logging to save disk space
    )
    
    # Validate the model
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50}")
    
    # Export the model (Optional: export to ONNX for deployment if needed)
    # success = model.export(format='onnx')
    # print(f"Model exported: {success}")
    print(f"Training complete. Best model saved to {model.trainer.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_model()
