import numpy as np
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict

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
        targets: List of target dictionaries (annotations)
        image_ids: List of image IDs
        
    Returns:
        coco_gt: Ground truth in COCO format
        coco_dt: Detections in COCO format
    """
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []
    
    # Create categories
    category_map = {}
    for i, cat in enumerate(set([ann["category_id"] for target in targets for ann in target["annotations"]])):
        category_map[cat] = i
        coco_gt["categories"].append({"id": i, "name": str(cat)})
    
    ann_id = 0
    for i, (target, image_id) in enumerate(zip(targets, image_ids)):
        # Add image info
        coco_gt["images"].append({
            "id": image_id,
            "width": target.get("width", 800),
            "height": target.get("height", 600)
        })
        
        # Add ground truth annotations
        for ann in target["annotations"]:
            x, y, w, h = ann["bbox"]
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_map[ann["category_id"]],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
        
        # Add predictions
        if i < len(predictions):
            pred = predictions[i]
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

def evaluate_detections(coco_gt, coco_dt):
    """
    Evaluate object detections using COCO metrics.
    
    Args:
        coco_gt: Ground truth in COCO format
        coco_dt: Detections in COCO format
        
    Returns:
        metrics: Dictionary of evaluation metrics
        pr_curves: Dictionary of precision-recall curves per category
    """
    # Save to temporary files
    with open("gt.json", "w") as f:
        json.dump(coco_gt, f)
    with open("dt.json", "w") as f:
        json.dump(coco_dt, f)
    
    # Initialize COCO ground truth and detections
    coco_gt = COCO("gt.json")
    coco_dt = coco_gt.loadRes("dt.json")
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
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
    for cat_id in coco_gt.getCatIds():
        precisions = coco_eval.eval["precision"][0, :, cat_id, 0, 2]  # IoU=0.5, all areas, max detections=100
        recalls = np.arange(0, 1.01, 0.01)
        pr_curves[cat_id] = (recalls, precisions)
    
    return metrics, pr_curves