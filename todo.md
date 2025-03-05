**** Masalah 1:
Faster RCNN, DEIM, dan YOLO punya format image dan targets yang berbeda, ini tercemin di train dimana:
DEIM dan faster RCNN membutuhkan: targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets] sedangkan yolo hanya:targets = targets.to(device)

**solusi**: Output _get_item dalam dataset yolo ubah ke format coco dan dalam YOLO class buat preprocess untuk menerima format targets seperti format COCO dan ubah ke YOLO.