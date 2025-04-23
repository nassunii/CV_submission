import os
import cv2
import yaml
import torch
import random
import numpy as np
from datetime import datetime
from models import YOLOvN
from models.YOLOvN import *
from pathlib import Path
'''
YOLOv9t 5e-5 & 0.15
'''


def augment_dataset(dataset_root: str, image_folder: str, label_folder: str, train_txt: str):
    img_dir = Path(dataset_root) / image_folder
    lbl_dir = Path(dataset_root) / label_folder
    img_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    if any(img_dir.rglob("*_flip.*")):
        print("[Offline 증강] 데이터가 이미 증강되었습니다.")
        return
    org_imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in img_ext]
    for p in org_imgs:
        flipped = cv2.flip(cv2.imread(str(p)), 1)
        flip_img = p.parent / f"{p.stem}_flip{p.suffix}"
        cv2.imwrite(str(flip_img), flipped)
        lbl_src = lbl_dir / f"{p.stem}.txt"
        lbl_dst = lbl_dir / f"{flip_img.stem}.txt"
        with open(lbl_src) as fr, open(lbl_dst, "w") as fw:
            for line in fr:
                cls, x, y, w, h = line.strip().split()
                fw.write(f"{cls} {1.0 - float(x):.6f} {y} {w} {h}\n")
    txt_files = list(Path(dataset_root).rglob(train_txt + "*.txt"))
    for txt_path in txt_files:
        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if any(line.endswith(ext) for ext in img_ext):
                stem, ext = os.path.splitext(line)
                new_lines.append(f"{stem}_flip{ext}")
        with open(txt_path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
    print(f"[Offline 증강] 증강 완료 — {len(org_imgs)}개 이미지 추가")

def submission_1_20223507(yaml_path, output_json_path):
    ###### can be modified (Only Hyperparameters, which can be modified in demo) ######
    data_config = load_yaml_config(yaml_path)
    ex_dict = {}
    offline_augment = True
    image_folder = "images"
    label_folder = "labels"
    train_txt = "train_iter_"
    epochs = 20
    batch_size = 16
    optimizer = 'AdamW'
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-5
    confidence = 0.15
    
    ###### can be modified (Only Models, which can't be modified in demo) ######
    from ultralytics import YOLO
    model_name = 'yolov9t'
    if offline_augment: augment_dataset(data_config["path"], image_folder, label_folder, train_txt)
    Experiments_Time = datetime.now().strftime("%y%m%d_%H%M%S")
    ex_dict['Iteration']  = int(yaml_path.split('.yaml')[0][-2:])
    image_size = 640
    output_dir ='tmp'
    optim_args = {'optimizer': optimizer, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    devices = [0]
    device = torch.device("cuda:"+str(devices[0])) if len(devices)>0 else torch.device("cpu")
    ex_dict['Experiment Time'] = Experiments_Time
    ex_dict['Epochs'] = epochs
    ex_dict['Batch Size'] = batch_size
    ex_dict['Device'] = device
    ex_dict['Optimizer'] = optimizer
    ex_dict['LR']=optim_args['lr']
    ex_dict['Weight Decay']=optim_args['weight_decay']
    ex_dict['Momentum']=optim_args['momentum']
    ex_dict['Image Size'] = image_size
    ex_dict['Output Dir'] = output_dir 
    Dataset_Name = yaml_path.split('/')[1]
    ex_dict['Dataset Name'] = Dataset_Name
    ex_dict['Data Config'] = yaml_path
    ex_dict['Number of Classes'] = data_config['nc']
    ex_dict['Class Names'] = data_config['names']
    control_random_seed(42)
    model = YOLO(f'{model_name}.yaml', verbose=False)
    os.makedirs(output_dir, exist_ok=True)
    ex_dict['Model Name'] = model_name
    ex_dict['Model']=model
    ex_dict = YOLOvN.train_model(ex_dict)
    test_images = get_test_images(data_config)
    results_dict = YOLOvN.detect_and_save_bboxes(ex_dict['Model'], test_images, confidence)
    save_results_to_file(results_dict, output_json_path)

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_test_images(config):
    test_path = config['test']
    root_path = config['path']

    test_path = os.path.join(root_path, test_path)
    
    if os.path.isdir(test_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    elif test_path.endswith('.txt'):
        with open(test_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        return image_paths

def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False 

def detect_and_save_bboxes(model, image_paths):
    results_dict = {}

    for img_path in image_paths:
        results = model(img_path, verbose=False, task='detect')
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

def save_results_to_file(results_dict, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"결과가 {output_path}에 저장되었습니다.")