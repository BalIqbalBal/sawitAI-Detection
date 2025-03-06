import numpy as np
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import seaborn as sns
import torch

def compute_confusion_matrix(coco_gt, coco_dt, iou_threshold=0.5):
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_ids = [cat['id'] for cat in categories]
    category_names = [cat['name'] for cat in categories]
    num_categories = len(category_ids)
    
    confusion_matrix = np.zeros((num_categories + 1, num_categories + 1), dtype=np.int32)
    
    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        
        gt_matched = [False] * len(gt_anns)
        
        for dt_ann in dt_anns:
            dt_cat = dt_ann['category_id']
            dt_cat_idx = category_ids.index(dt_cat) if dt_cat in category_ids else -1
            
            if dt_cat_idx == -1:
                continue
                
            max_iou = -1
            max_gt_idx = -1
            
            for gt_idx, gt_ann in enumerate(gt_anns):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = compute_iou(dt_ann['bbox'], gt_ann['bbox'])
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx != -1:
                gt_cat = gt_anns[max_gt_idx]['category_id']
                gt_cat_idx = category_ids.index(gt_cat)
                gt_matched[max_gt_idx] = True
                
                if dt_cat_idx == gt_cat_idx:
                    confusion_matrix[gt_cat_idx, dt_cat_idx] += 1
                else:
                    confusion_matrix[gt_cat_idx, dt_cat_idx] += 1
            else:
                confusion_matrix[num_categories, dt_cat_idx] += 1
        
        for gt_idx, gt_ann in enumerate(gt_anns):
            if not gt_matched[gt_idx]:
                gt_cat_idx = category_ids.index(gt_ann['category_id'])
                confusion_matrix[gt_cat_idx, num_categories] += 1
    
    return confusion_matrix, category_names

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x1_inter = max(x1, x2)
    y1_inter = max(y1, y2)
    x2_inter = min(x1 + w1, x2 + w2)
    y2_inter = min(y1 + h1, y2 + h2)
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area

def plot_confusion_matrix(confusion_matrix, category_names, filename="confusion_matrix.png"):
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
    with open("gt.json", "w") as f:
        json.dump(coco_gt, f)
    with open("dt.json", "w") as f:
        json.dump(coco_dt, f)
    
    coco_gt_obj = COCO("gt.json")
    
    if not coco_dt:
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
    
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metrics = {
        "mAP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_max1": coco_eval.stats[6],
        "AR_max10": coco_eval.stats[7],
        "AR_max100": coco_eval.stats[8]
    }
    
    pr_curves = {}
    for cat_id in coco_gt_obj.getCatIds():
        precisions = coco_eval.eval["precision"][0, :, cat_id, 0, 2]
        recalls = np.arange(0, 1.01, 0.01)
        pr_curves[cat_id] = (recalls, precisions)
    
    confusion_matrix, category_names = compute_confusion_matrix(coco_gt_obj, coco_dt_obj)
    confusion_matrix_file = plot_confusion_matrix(confusion_matrix, category_names)
    
    return metrics, pr_curves, confusion_matrix_file

def convert_to_coco_format(predictions, targets, image_ids, image_sizes):
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []

    # Process categories
    category_ids = set()
    for target in targets:
        if isinstance(target, dict) and "labels" in target:
            category_ids.update(target["labels"].tolist())
    
    # Create category mapping (COCO requires category IDs to start at 1)
    category_map = {cat: i+1 for i, cat in enumerate(sorted(category_ids))}
    coco_gt["categories"] = [{"id": v, "name": str(k)} for k, v in category_map.items()]

    ann_id = 1  # Annotation IDs should start at 1

    # Process images and annotations
    for i, (image_id, (width, height)) in enumerate(zip(image_ids, image_sizes)):
        # Add image metadata with actual dimensions
        coco_gt["images"].append({
            "id": image_id,
            "width": width,
            "height": height
        })

        # Process ground truth
        target = targets[i]
        if isinstance(target, dict) and "boxes" in target and "labels" in target:
            for box, label in zip(target["boxes"], target["labels"]):
                x, y, w, h = box.tolist()  # Convert tensor to list
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_map[label.item()],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

        # Process predictions
        if i < len(predictions):
            pred = predictions[i]
            if isinstance(pred, dict) and all(k in pred for k in ["boxes", "scores", "labels"]):
                for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                    x, y, w, h = box.tolist()
                    coco_dt.append({
                        "image_id": image_id,
                        "category_id": category_map.get(label.item(), -1),
                        "bbox": [x, y, w, h],
                        "score": score.item(),
                        "area": w * h
                    })

    return coco_gt, coco_dt