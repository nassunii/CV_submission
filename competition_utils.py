import os
import cv2
import time
import yaml
import json
import math
import psutil
import random
import numpy as np
import pandas as pd
import natsort
from datetime import datetime
from scipy.stats import ttest_rel, rankdata
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from shapely.geometry import box, Polygon
from matplotlib.collections import PatchCollection

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_test_image_list(test_list_file):
    image_paths = []
    with open(test_list_file, 'r') as f:
        for line in f:
            image_path = line.strip()
            if image_path:  # 빈 줄 무시
                image_paths.append(image_path)
    return image_paths

def load_detection_results(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        detection_results = json.load(f)
    return detection_results

def load_yolo_labels(label_path, img_width, img_height):
   
    bboxes = []
    classes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # 클래스 ID, x_center, y_center, width, height
                # YOLO 형식: 클래스_ID, x_center, y_center, width, height (모두 정규화된 값)
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height
                
                bboxes.append([x1, y1, x2, y2])
                classes.append(class_id)

    return bboxes, classes

def bboxes_to_union_mask(bboxes, image_shape=(640, 640)):
    """
    bboxes: list of [x1, y1, x2, y2] (float)
    image_shape: tuple (H, W)
    returns: uint8 mask of shape image_shape, with 1 inside any bbox, 0 elsewhere
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    
    for x1, y1, x2, y2 in bboxes:
        # 실수 좌표를 정수 인덱스로 변환
        x1_i = int(math.floor(x1))
        y1_i = int(math.floor(y1))
        x2_i = int(math.ceil(x2))
        y2_i = int(math.ceil(y2))
        # 이미지 경계 내로 클리핑
        x1_i = max(0, min(W, x1_i))
        x2_i = max(0, min(W, x2_i))
        y1_i = max(0, min(H, y1_i))
        y2_i = max(0, min(H, y2_i))
        # 마스크 채우기
        mask[y1_i:y2_i, x1_i:x2_i] = 1

    return mask

def eval_masks(pred_mask: np.ndarray, gt_mask: np.ndarray):

    assert pred_mask.shape == gt_mask.shape, "pred_mask와 gt_mask의 shape이 같아야 합니다."
    
    # bool 배열로 변환
    pred = (pred_mask > 0)
    gt   = (gt_mask   > 0)
    
    # 교집합, 합집합, TP/FP/FN 계산
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    tp = intersection
    fp = pred.sum() - tp
    fn = gt.sum()   - tp
    
    # 계산
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    iou       = tp / (union + 1e-6)
    dice      = 2 * tp / (pred.sum() + gt.sum() + 1e-6)
    
    return float(iou), float(dice), float(precision),float(recall)


def visualize_and_save_comparison(image_path, detection_results, ground_truth_bboxes, 
                                 ground_truth_classes, iou, output_dir, class_names=None):
    """감지 결과와 Ground Truth를 시각화하고 IoU를 표시하여 저장하는 함수"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # BGR에서 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기
    height, width = image.shape[:2]
    
    # 플롯 생성
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # 색상 설정
    detection_color = 'red'  # 감지 결과는 빨간색
    gt_color = 'green'  # Ground Truth는 녹색색
    
    # 랜덤 색상 생성 함수
    def get_random_color():
        return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    
    # 클래스별 색상 맵 생성
    if class_names:
        class_colors = {i: get_random_color() for i in range(len(class_names))}
    else:
        class_colors = {}
        
    # 각 감지 결과 바운딩 박스 그리기
    for i, detection in enumerate(detection_results):
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]
        class_name = detection["class_name"] if "class_name" in detection else f"Class {class_id}"
        
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=detection_color, facecolor='none')
        ax.add_patch(rect)
        
        # 클래스 이름과 신뢰도 표시
        label = f"{class_name}: {confidence:.2f}"
        ax.text(x1, y1-5, label, color=detection_color, fontsize=8, backgroundcolor="white")
    
    # 각 Ground Truth 바운딩 박스 그리기
    for i, (bbox, class_id) in enumerate(zip(ground_truth_bboxes, ground_truth_classes)):
        x1, y1, x2, y2 = bbox
        
        # 클래스별 색상 사용 (있는 경우)
        edge_color = class_colors.get(class_id, gt_color)
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=edge_color, facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # 클래스 이름 표시
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        ax.text(x1, y2+15, class_name, color=edge_color, fontsize=8, backgroundcolor="white")
    
    # IoU 값 표시
    ax.text(10, 30, f"IoU: {iou:.4f}", color='black', fontsize=12, backgroundcolor="white", weight='bold')
    
    # 범례 추가
    import matplotlib.lines as mlines
    
    det_line = mlines.Line2D([], [], color=detection_color, marker='s', linestyle='-', markersize=10, label='Detection')
    gt_line = mlines.Line2D([], [], color=gt_color, marker='s', linestyle='--', markersize=10, label='Ground Truth')
    
    ax.legend(handles=[det_line, gt_line], loc='upper right')
    
    # 축 제거
    plt.axis('off')
    
    # 파일명 생성 및 저장 (IoU 값을 파일명 앞에 추가)
    # IoU 값을 4자리 숫자로 변환 (예: 0.8546 -> 0854)
    iou_prefix = f"{iou:.4f}".replace(".", "")
    output_filename = f"iou_{iou_prefix}_file_name_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    return output_path

def eval_and_vis(yaml_path, det_result_path, labels_dir, image_level_result_path, vis_output_dir, vis=False):
    config = load_yaml_config(yaml_path)
    class_names = config.get('names', None)
    test_path = os.path.join(config.get('path', ''), config.get('test', ''))
    test_image_list = read_test_image_list(test_path)
    
    detection_results = load_detection_results(det_result_path)
    if vis==True:
        os.makedirs(vis_output_dir, exist_ok=True)
    performances = {}
    
    for image_path in test_image_list:
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        detections = detection_results[image_path]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, f"{image_name}.txt")
        detected_bboxes = [detection["bbox"] for detection in detections]
        ground_truth_bboxes, ground_truth_classes = load_yolo_labels(label_path, img_width, img_height)
        detected_union_bbox = bboxes_to_union_mask(detected_bboxes)
        ground_truth_union_bbox = bboxes_to_union_mask(ground_truth_bboxes)
        iou, dice, precision, recall = eval_masks(detected_union_bbox, ground_truth_union_bbox)
        # visualize_union_masks(image_path, detected_union_bbox, ground_truth_union_bbox)
        performances[image_path] = {
            "IoU": iou,
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "detected_boxes_count": len(detected_bboxes),
            "ground_truth_boxes_count": len(ground_truth_bboxes)
        }
        if vis==True:
            vis_path = visualize_and_save_comparison(
                image_path, 
                detections, 
                ground_truth_bboxes, 
                ground_truth_classes,
                iou, 
                vis_output_dir,
                class_names
            )
    
    with open(image_level_result_path, 'w', encoding='utf-8') as f:
        json.dump(performances, f, ensure_ascii=False, indent=4)
    if performances:
        evals = ["IoU", "Dice", "Precision", "Recall"]
        stats = {}
    
        for m in evals:
            values = [res[m] for res in performances.values()]
            avg = sum(values) / len(values)
            std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
            stats[m] = {"avg": avg, "std": std}
        return stats
    return 0

def reshape_df(input_csv_path, keep_columns, column_groups, groups_name, value_name):
    # Load CSV file
    df = pd.read_csv(input_csv_path, encoding='cp949')
    # Drop rows with missing values in specified columns
    df = df.dropna(subset=keep_columns + column_groups)
    # Preserve selected columns and reshape DataFrame
    df= df.melt(id_vars=keep_columns, value_vars=column_groups, var_name=groups_name, value_name=value_name)
    if "Iteration" in df.columns:
        df = df.copy()
        df["Iteration"] = pd.to_numeric(df["Iteration"], errors='coerce').astype('Int64')
    return df

def filter_dataframe_by_keys(df, row, column, rename_row, rename_col):
    # row와 column 값이 각각 rename_dict_row 및 rename_dict_col의 키에 해당하는 경우만 필터링
    filtered_df = df[
        df[row].isin(rename_row) & 
        df[column].isin(rename_col)
    ]
    return filtered_df

def filter_and_transform_formal_name(df, row, column, rename_row, rename_col, null_column):
    if isinstance(rename_col, dict) and isinstance(rename_row, dict):
        df = filter_dataframe_by_keys(df, row, column, list(rename_row.keys()), list(rename_col.keys()))
        df = transform_column_values(df, column, rename_col)
        df = transform_column_values(df, row, rename_row)
        null_column = rename_col[null_column]
    elif isinstance(rename_col, list) and isinstance(rename_row, list):
        df = filter_dataframe_by_keys(df, row, column, rename_row, rename_col)
    else:
        print('filter_and_transform_formal_name Error')
    return df, null_column
    
def transform_column_values(df, column, conversion_dict):
    """
    주어진 DataFrame의 특정 컬럼에 대해 값들을 사전에 정의된 딕셔너리를 사용하여 변환하는 함수.

    Parameters:
    df (pd.DataFrame): 변환할 DataFrame
    column (str): 변환할 대상 컬럼 이름
    conversion_dict (dict): 키-값 쌍으로 정의된 변환 딕셔너리

    Returns:
    pd.DataFrame: 변환된 DataFrame
    """
    if column in df.columns:
        df = df.copy()
        df[column] = df[column].map(conversion_dict).fillna(df[column])  # 일치하지 않는 값은 원래 값 유지
    return df

def reorder_dataframe(df, new_column_order, new_row_order) :
    if isinstance(new_column_order, dict) and isinstance(new_row_order, dict):
        new_column_order = list(new_column_order.values())
        new_row_order = list(new_row_order.values())
    elif isinstance(new_column_order, list) and isinstance(new_row_order, list):
        pass
    else:
        print('reorder_dataframe Error')
    if all(col in df.columns for col in new_column_order):
        df = df[new_column_order]
    else:
        raise ValueError("new_column_order에 포함된 모든 열이 DataFrame에 존재하지 않습니다.")

    # 행 순서 재정렬
    if len(df) == len(new_row_order):
        df = df.reindex(new_row_order)
    else:
        raise ValueError("new_row_order의 길이가 DataFrame의 행 수와 일치하지 않습니다.")

    return df

def get_unique_rows(df, row):
    """Get unique rows sorted using natsort."""
    return natsort.natsorted(df[row].unique())

def extract_values_by_column(row_subset, column, unique_columns):
    """Extract values for each unique column."""
    return {c: row_subset[row_subset[column] == c]['value'].tolist() for c in unique_columns}

def perform_paired_t_tests(null_values, values_by_column, unique_columns, null_column, significance_levels, common_length):
    """Perform paired t-tests and determine significance annotations."""
    significance_annotations = {}
    for c in unique_columns:
        if c != null_column and len(values_by_column[c]) >= common_length:
            t_stat, p_val = ttest_rel(null_values[:common_length], values_by_column[c][:common_length])
            # Determine significance annotation with △ or ▼
            annotation = ''
            for i, level in enumerate(significance_levels):
                if p_val < level:
                    if np.mean(values_by_column[c][:common_length]) > np.mean(null_values[:common_length]):
                        annotation = '△' * (i + 1)
                    else:
                        annotation = '▼' * (i + 1)
            significance_annotations[c] = annotation
    return significance_annotations

def calculate_ranks(value_matrix_dict, ranking_order='asc'):
    """
    Calculate ranks across different columns for each index (iteration).
    
    Args:
        value_matrix_dict (dict): Dictionary where keys are column names and values are lists of values.
        ranking_order (str): 'asc' for ascending, 'desc' for descending ranking.

    Returns:
        dict: Dictionary with average ranks for each column.
    """
    # Convert dictionary to matrix (list of rows where each row is one iteration's values across columns)
    columns = list(value_matrix_dict.keys())
    matrix = np.array([value_matrix_dict[col] for col in columns]).T  # Transpose to make rows represent iterations

    # Calculate ranks for each row
    ranks_per_row = [
        rankdata(row if ranking_order == 'asc' else [-val for val in row])
        for row in matrix
    ]
    
    # Average ranks across rows for each column
    avg_ranks = np.mean(ranks_per_row, axis=0)
    
    # Map average ranks back to column names
    return {columns[i]: avg_ranks[i] for i in range(len(columns))}
def apply_custom_formatting(mean_value, std_value, annotation, decimal_places, custom_fmt_template):

    plus_minus = "±"

    formatted_text = custom_fmt_template.format(
        mean_fmt=f"{mean_value:.{decimal_places}f}",
        std_fmt=f"{std_value:.{decimal_places}f}",
        significance=annotation,
    ).replace("±", plus_minus)

    return formatted_text

def analyze_row_column_combinations_with_ranking(df, row, column, unique_columns, ranking_order_list, null_column, significance_levels, decimal_places=2, custom_fmt_template='{underline_prefix}{bold_prefix}{mean_fmt}±{std_fmt} {significance}{bold_suffix}{underline_suffix}'):
    """Main function to analyze row-column combinations with ranking."""
    unique_rows = get_unique_rows(df, row)
    results = []

    for idx, r in enumerate(unique_rows):
        ranking_order = ranking_order_list[idx] if idx < len(ranking_order_list) else 'asc'
        row_subset = df[df[row] == r]
        
        values_by_column = extract_values_by_column(row_subset, column, unique_columns)
        common_length = min(len(values) for values in values_by_column.values() if len(values) > 0)
        if common_length == 0:
            continue  # Skip if no valid data is present

        value_matrix = [[values_by_column[c][idx] for c in unique_columns if len(values_by_column[c]) > idx] for idx in range(common_length)]
        
        ranked_results = calculate_ranks(values_by_column, ranking_order)
        null_values = values_by_column.get(null_column, [])
        significance_annotations = perform_paired_t_tests(null_values, values_by_column, unique_columns, null_column, significance_levels, common_length)

        # Calculate mean values for determining max mean per row
        mean_values = {c: np.mean(values_by_column[c]) for c in unique_columns if values_by_column[c]}
        sorted_mean_values = sorted(mean_values.items(), key=lambda item: item[1], reverse=True)
        max_mean_value = sorted_mean_values[0][1] if mean_values else None
        second_max_mean_value = sorted_mean_values[1][1] if len(sorted_mean_values) > 1 else None

        row_results = {}
        for c in unique_columns:
            ranks = ranked_results[c]
            if ranks:
                mean_rank = np.mean(ranks)
                mean_value = np.mean(values_by_column[c])
                std_value = np.std(values_by_column[c])
                annotation = significance_annotations.get(c, '')
                # Determine if bold or underline should be applied
                formatted_result = apply_custom_formatting(mean_value, std_value, annotation, decimal_places, custom_fmt_template)
                row_results[c] = formatted_result
            else:
                row_results[c] = "N/A"
        results.append(row_results)

    final_df = pd.DataFrame(results, index=unique_rows, columns=unique_columns)

    return final_df

def make_tables(input_csv_path, keep_columns, keep_measures, reduction, row, column, null_column, custom_fmt_template,
                significance_levels,decimal_places, transpose
               ):

    df = reshape_df(input_csv_path, keep_columns, keep_measures, groups_name='Measure Type', value_name='value')
    rename_row = dict(zip(df[row].unique(), df[row].unique()))
    rename_col = dict(zip(df[column].unique(), df[column].unique()))
    ranking_order_list = ['desc'] * len(rename_row)
    
    second_df, null_column = filter_and_transform_formal_name(df, row, column, rename_row, rename_col, null_column)
    
    third_df = second_df
    fourth_df = analyze_row_column_combinations_with_ranking(
        third_df, 
        row=row, 
        column=column, 
        unique_columns=third_df[column].unique(), 
        ranking_order_list=ranking_order_list,
        null_column=null_column, 
        significance_levels=significance_levels, 
        decimal_places=decimal_places, 
        custom_fmt_template=custom_fmt_template,
    )
    fourth_df = reorder_dataframe(fourth_df, rename_col, rename_row)
    fourth_df = fourth_df.T if transpose else fourth_df
    output_csv_path = input_csv_path.replace('.csv', '_agg.csv')
    
    fourth_df.to_csv(output_csv_path, index=True, encoding='UTF-8')
    print(f"CSV 파일로 저장 완료: {output_csv_path}")
