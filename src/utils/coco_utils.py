import numpy as np
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import seaborn as sns
import torch

def compute_confusion_matrix(coco_gt, coco_dt, iou_threshold=0.5):
    """
    Compute confusion matrix for object detection.
    
    Args:
        coco_gt: COCO ground truth object
        coco_dt: COCO detection object
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        confusion_matrix: Numpy array of the confusion matrix
        category_names: List of category names
    """
    # Get category information
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_ids = [cat['id'] for cat in categories]
    category_names = [cat['name'] for cat in categories]
    num_categories = len(category_ids)
    
    # Initialize confusion matrix (rows=ground truth, cols=predictions)
    confusion_matrix = np.zeros((num_categories + 1, num_categories + 1), dtype=np.int32)
    
    # Iterate over all images
    for img_id in coco_gt.getImgIds():
        # Get ground truth annotations for the image
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        # Get detection annotations for the image
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        
        # Create matrices for IoU calculation
        gt_matched = [False] * len(gt_anns)
        
        # Calculate IoU between each detection and ground truth
        for dt_idx, dt_ann in enumerate(dt_anns):
            dt_cat_idx = category_ids.index(dt_ann['category_id']) if dt_ann['category_id'] in category_ids else -1
            
            # Skip if category is not recognized
            if dt_cat_idx == -1:
                continue
                
            max_iou = -1
            max_gt_idx = -1
            
            # Find the best matching ground truth for this detection
            for gt_idx, gt_ann in enumerate(gt_anns):
                if gt_matched[gt_idx]:
                    continue
                    
                # Calculate IoU
                iou = compute_iou(dt_ann['bbox'], gt_ann['bbox'])
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # If we found a match above the threshold
            if max_iou >= iou_threshold and max_gt_idx != -1:
                gt_cat_idx = category_ids.index(gt_anns[max_gt_idx]['category_id'])
                gt_matched[max_gt_idx] = True
                
                # True positive: correct classification
                if dt_cat_idx == gt_cat_idx:
                    confusion_matrix[gt_cat_idx, dt_cat_idx] += 1
                # Misclassification: wrong class
                else:
                    confusion_matrix[gt_cat_idx, dt_cat_idx] += 1
            else:
                # False positive: detection without matching ground truth
                confusion_matrix[num_categories, dt_cat_idx] += 1
        
        # Count unmatched ground truths as false negatives
        for gt_idx, gt_ann in enumerate(gt_anns):
            if not gt_matched[gt_idx]:
                gt_cat_idx = category_ids.index(gt_ann['category_id'])
                confusion_matrix[gt_cat_idx, num_categories] += 1
    
    return confusion_matrix, category_names

def compute_iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes in [x, y, width, height] format.
    """
    # Convert from [x, y, width, height] to [x1, y1, x2, y2]
    x1_1, y1_1 = bbox1[0], bbox1[1]
    x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    
    x1_2, y1_2 = bbox2[0], bbox2[1]
    x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    
    # Compute intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Compute IoU
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area

def plot_confusion_matrix(confusion_matrix, category_names, filename="sementara/confusion_matrix.png"):
    """
    Plot confusion matrix and save it to a file.
    
    Args:
        confusion_matrix: Numpy array of the confusion matrix
        category_names: List of category names
        filename: Output filename
    """
    # Add "Background" for false positives and false negatives
    extended_names = category_names + ["Background"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=extended_names, yticklabels=extended_names)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix for Object Detection')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

def evaluate_detections(coco_gt, coco_dt):
    """
    Evaluate object detections using COCO metrics.
    
    Args:
        coco_gt: Ground truth in COCO format
        coco_dt: Detections in COCO format
        
    Returns:
        metrics: Dictionary of evaluation metrics
        pr_curves: Dictionary of precision-recall curves per category
        confusion_matrix_file: Filename of the confusion matrix plot
    """
    # Save to temporary files
    with open("gt.json", "w") as f:
          json.dump(coco_gt, f)
    with open("dt.json", "w") as f:
        json.dump(coco_dt, f)
    
    coco_gt_obj = COCO("gt.json")
    
    # Handle empty detections
    if not coco_dt:
        print("Warning: No valid detections found. Returning zero metrics.")
        return {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR_max1": 0.0,
            "AR_max10": 0.0,
            "AR_max100": 0.0
        }, {}, None
    
    coco_dt_obj = coco_gt_obj.loadRes("dt.json")
    
    # Initialize COCO ground truth and detections
    coco_gt_obj = COCO("gt.json")
    coco_dt_obj = coco_gt_obj.loadRes("dt.json")
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        "mAP": coco_eval.stats[0],  # mAP at IoU=0.5:0.95
        "AP50": coco_eval.stats[1],  # mAP at IoU=0.5
        "AP75": coco_eval.stats[2],  # mAP at IoU=0.75
        "AP_small": coco_eval.stats[3],  # mAP for small objects
        "AP_medium": coco_eval.stats[4],  # mAP for medium objects
        "AP_large": coco_eval.stats[5],  # mAP for large objects
        "AR_max1": coco_eval.stats[6],  # AR given 1 detection per image
        "AR_max10": coco_eval.stats[7],  # AR given 10 detections per image
        "AR_max100": coco_eval.stats[8]  # AR given 100 detections per image
    }
    
    # Extract precision-recall curves (one per category)
    pr_curves = {}
    for cat_id in coco_gt_obj.getCatIds():
        precisions = coco_eval.eval["precision"][0, :, cat_id, 0, 2]  # IoU=0.5, all areas, max detections=100
        recalls = np.arange(0, 1.01, 0.01)
        pr_curves[cat_id] = (recalls, precisions)
    
    # Compute and plot confusion matrix
    confusion_matrix, category_names = compute_confusion_matrix(coco_gt_obj, coco_dt_obj, iou_threshold=0.5)
    confusion_matrix_file = plot_confusion_matrix(confusion_matrix, category_names)
    
    return metrics, pr_curves, confusion_matrix_file

def plot_precision_recall_curve(precisions, recalls, ap, filename):
    """
    Plot precision-recall curve for a category and save to file.
    
    Args:
        precisions: Array of precision values
        recalls: Array of recall values
        ap: Average precision value
        filename: Output filename for the plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP: {ap:.4f})")
    plt.fill_between(recalls, precisions, alpha=0.2)
    plt.savefig(filename)
    plt.close()

def convert_to_coco_format(predictions, targets, image_ids):
    """
    Convert model predictions and targets to COCO format for evaluation.
    
    Args:
        predictions: List of prediction dictionaries (boxes, scores, labels)
        targets: List of target dictionaries (boxes, labels)
        image_ids: List of image IDs
        
    Returns:
        coco_gt: Ground truth in COCO format
        coco_dt: Detections in COCO format
    """
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []

    # Ensure targets is a list of dictionaries
    if isinstance(targets, torch.Tensor):
        # Convert tensor to list of dictionaries if necessary
        targets = [{"boxes": targets[:, :4], "labels": targets[:, 4]}]

    # Pastikan targets memiliki category_id dengan mengambilnya dari 'labels'
    category_ids = set()
    for target in targets:
        if isinstance(target, dict) and "labels" in target:
            category_ids.update(target["labels"].tolist())

    category_map = {cat: i for i, cat in enumerate(sorted(category_ids))}
    for cat, idx in category_map.items():
        coco_gt["categories"].append({"id": idx, "name": str(cat)})

    ann_id = 0
    for i, (target, image_id) in enumerate(zip(targets, image_ids)):
        # Tambahkan informasi gambar
        coco_gt["images"].append({
            "id": image_id,
            "width": 800,  # Gantilah dengan ukuran gambar sebenarnya jika diketahui
            "height": 600
        })

        # Konversi ground truth dari 'boxes' dan 'labels'
        if isinstance(target, dict) and "boxes" in target and "labels" in target:
            for box, label in zip(target["boxes"], target["labels"]):
                x, y, w, h = box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_map.get(label.item(), 0),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

        # Konversi prediksi ke COCO format
        if i < len(predictions):
            pred = predictions[i]
            if isinstance(pred, dict) and "boxes" in pred and "labels" in pred and "scores" in pred:
                for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                    x, y, w, h = box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()
                    coco_dt.append({
                        "image_id": image_id,
                        "category_id": category_map.get(label.item(), 0),
                        "bbox": [x, y, w, h],
                        "score": score.item(),
                        "area": w * h
                    })

    return coco_gt, coco_dt