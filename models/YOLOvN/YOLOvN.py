import os
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from ultralytics import settings
settings.update({'datasets_dir': './'})

def train_model(ex_dict):
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    task = f"{ex_dict['Experiment Time']}_Train"

    ex_dict['Train Results'] = ex_dict['Model'].train(
        model = f"{ex_dict['Model Name']}.yaml",
        name=name,
        data=ex_dict['Data Config'] ,
        epochs=ex_dict['Epochs'],
        imgsz=ex_dict['Image Size'],
        batch=ex_dict['Batch Size'],
        patience=20,
        save=True,
        device=ex_dict['Device'],
        exist_ok=True,
        verbose=False,
        optimizer=ex_dict['Optimizer'],
        lr0=ex_dict['LR'],  
        weight_decay = ex_dict['Weight Decay'],
        momentum = ex_dict['Momentum'],
        pretrained=False,
        amp=False,
        task = task,
        project =f"{ex_dict['Output Dir']}",
    )
    pt_path = f"{ex_dict['Output Dir']}/{name}/weights/best.pt"
    ex_dict['PT path'] = pt_path
    ex_dict['Model'].load(pt_path)
    return ex_dict

def detect_and_save_bboxes(model, image_paths, conf):
    results_dict = {}
    for img_path in image_paths:
        results = model(img_path, verbose=False, task='detect', conf=conf)
        img_results = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                img_results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results
    return results_dict