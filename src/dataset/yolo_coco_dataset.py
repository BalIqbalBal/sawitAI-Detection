from pycocotools.coco import COCO
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input

class YoloCOCODataset(Dataset):
    def __init__(self, image_dir, coco_annotation_path, input_shape, num_classes, epoch_length, 
                 mosaic=True, mixup=True, mosaic_prob=0.5, mixup_prob=0.5, train=True, special_aug_ratio=0.7, 
                 transform=None):
        super(YoloCOCODataset, self).__init__()
        self.coco = COCO(coco_annotation_path)
        self.image_dir = image_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.epoch_now = -1
        self.bbox_attrs = 5 + num_classes

        self.transform = transform

        # Load COCO annotations
        self.load_coco_annotations()
        self.length = len(self.annotation_lines)

    def load_coco_annotations(self):
        """Convert COCO annotations to YOLO format"""
        self.annotation_lines = []
        img_ids = self.coco.getImgIds()
        
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            image_path = os.path.join(self.image_dir, img_info['file_name'])
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            boxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                class_id = ann['category_id'] - 1  # Convert to 0-based index
                x1 = float(x)
                y1 = float(y)
                x2 = float(x + w)
                y2 = float(y + h)
                boxes.append([x1, y1, x2, y2, class_id])
            
            if boxes:
                # Format: "path/to/img.jpg x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ..."
                box_strings = [','.join(map(str, box)) for box in boxes]
                line = f"{image_path} {' '.join(box_strings)}"
                self.annotation_lines.append(line)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # Apply transforms (if any)
        if self.transform:
            orig_w, orig_h = image.size
            image = self.transform(image)

            new_h, new_w = image.shape[1:]  # Transformed image dimensions
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, [0, 2]] *= scale_x  # Scale x-coordinates
            boxes[:, [1, 3]] *= scale_y  # Scale y-coordinates
        

        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = np.random.choice(self.annotation_lines, 3, replace=False)
            lines = np.append(lines, self.annotation_lines[index])
            np.random.shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
            if self.mixup and self.rand() < self.mixup_prob:
                line = np.random.choice(self.annotation_lines)
                image2, box2 = self.get_random_data(line, self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image2, box2)
        else:
           #image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
           pass
        
        # Preprocess image
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        
        # Format labels for COCO
        labels = np.zeros((len(box), 6))
        if len(box) > 0:
            # Normalize coordinates to [0,1]
            box[:, [0,2]] /= self.input_shape[1]  # x coordinates
            box[:, [1,3]] /= self.input_shape[0]  # y coordinates
            
            # Convert to [center_x, center_y, width, height]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]  # width, height
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2  # center coordinates
            
            labels[:, 1] = box[:, -1]  # Class IDs
            labels[:, 2:] = box[:, :4]  # Box coordinates

        # Convert to COCO format [x_min, y_min, width, height]
        if len(box) > 0:
            # Extract normalized center_x, center_y, width, height
            boxes = labels[:, 2:]
            class_ids = labels[:, 1].astype(np.int64)
            
            # Denormalize and convert to xywh
            x_min = (boxes[:, 0] - boxes[:, 2]/2) * self.input_shape[1]
            y_min = (boxes[:, 1] - boxes[:, 3]/2) * self.input_shape[0]
            width = boxes[:, 2] * self.input_shape[1]
            height = boxes[:, 3] * self.input_shape[0]
            
            boxes_coco = np.stack([x_min, y_min, width, height], axis=1)
            labels_coco = class_ids
        else:
            boxes_coco = np.zeros((0, 4), dtype=np.float32)
            labels_coco = np.zeros((0), dtype=np.int64)

        target = {
            "boxes": boxes_coco,
            "labels": labels_coco,
        }

        return image, target

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape

        # Parse annotation with float coordinates
        box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

        if not random:
            # Rescale without augmentation
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            
            if len(box) > 0:
                box[:, [0,2]] = box[:, [0,2]]*scale + dx
                box[:, [1,3]] = box[:, [1,3]]*scale + dy
                box[:, 0:2] = np.maximum(box[:, 0:2], 0)
                box[:, 2] = np.minimum(box[:, 2], w)
                box[:, 3] = np.minimum(box[:, 3], h)
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
            return image_data, box

        # Apply data augmentation
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # Place image with random offset
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # Random horizontal flip
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Apply HSV color augmentation
        image_data = np.array(image, dtype=np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        lut_hue = ((np.arange(0, 256, dtype=np.float32) * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(np.arange(0, 256, dtype=np.float32) * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(np.arange(0, 256, dtype=np.float32) * r[2], 0, 255).astype(np.uint8)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # Adjust bounding boxes
        if len(box) > 0:
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip:
                box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2] = np.maximum(box[:, 0:2], 0)
            box[:, 2] = np.minimum(box[:, 2], w)
            box[:, 3] = np.minimum(box[:, 3], h)
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
        return image_data, box

    def get_random_data_with_Mosaic(self, annotation_lines, input_shape, jitter=0.3):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        image_datas = []
        box_datas = []
        
        for i, line in enumerate(annotation_lines):
            line_content = line.split()
            image = Image.open(line_content[0])
            image = cvtColor(image)
            iw, ih = image.size
            
            # Parse annotation with float coordinates
            box = np.array([np.array(list(map(float, box.split(',')))) for box in line_content[1:]])
            
            # Random horizontal flip
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]
            
            # Random scaling and aspect ratio distortion
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            
            # Position images in mosaic grid
            if i == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif i == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif i == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif i == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            
            # Adjust bounding boxes
            if len(box) > 0:
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2] = np.maximum(box[:, 0:2], 0)
                box[:, 2] = np.minimum(box[:, 2], w)
                box[:, 3] = np.minimum(box[:, 3], h)
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
            
            image_datas.append(image_data)
            box_datas.append(box)

        # Combine mosaic images
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # Apply HSV color augmentation
        r = np.random.uniform(-1, 1, 3) * [0.1, 0.7, 0.4] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        lut_hue = ((np.arange(0, 256) * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(np.arange(0, 256) * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(np.arange(0, 256) * r[2], 0, 255).astype(np.uint8)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # Merge bounding boxes from all mosaic parts
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        return new_image, new_boxes

    def merge_bboxes(self, boxes, cutx, cuty):
        merged_boxes = []
        for i in range(4):
            for box in boxes[i]:
                x1, y1, x2, y2, cls = box
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    x2 = min(x2, cutx)
                    y2 = min(y2, cuty)
                elif i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    x2 = min(x2, cutx)
                    y1 = max(y1, cuty)
                elif i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    x1 = max(x1, cutx)
                    y1 = max(y1, cuty)
                elif i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    x1 = max(x1, cutx)
                    y2 = min(y2, cuty)
                merged_boxes.append([x1, y1, x2, y2, cls])
        return np.array(merged_boxes)

    def get_random_data_with_MixUp(self, image1, box1, image2, box2):
        new_image = (np.array(image1, dtype=np.float32) * 0.5 + 
                     np.array(image2, dtype=np.float32) * 0.5)
        new_boxes = np.concatenate([box1, box2], axis=0)
        return new_image.astype(np.uint8), new_boxes

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a