import os

from ultralytics import YOLO

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1" 

model = YOLO("yolo11n.pt")  

results = model.train(data="src/train/leaf_classification_data/data.yaml", 
                      epochs=2,
                      imgsz=640, 
                      device=[0,1],
                      batch=32, 
                      save=True, 
                      workers=8, 
                      seed=42, 
                      optimizer="AdamW", 
                      lr0=0.000005)