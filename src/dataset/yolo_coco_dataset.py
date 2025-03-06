import os
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset
from random import sample, shuffle

from utils.utils import cvtColor, preprocess_input

class YoloCOCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, input_shape, num_classes, epoch_length, 
                 mosaic=True, mixup=True, mosaic_prob=0.5, mixup_prob=0.5, train=True, special_aug_ratio=0.7, transform=None):
        super(YoloCOCODataset, self).__init__()
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.transform = transform

        self.epoch_now = -1
        self.image_ids = self._filter_valid_image_ids()
        self.length = len(self.image_ids)

    def _filter_valid_image_ids(self):
        valid_image_ids = []
        for image_id in self.coco.imgs.keys():
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.root_dir, image_info["file_name"])

            if os.path.exists(image_path) and self._is_valid_image(image_path):
                valid_image_ids.append(image_id)
            else:
                print(f"Invalid or missing image: {image_path} (Image ID: {image_id})")
        return valid_image_ids

    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image_id = self.image_ids[index]

        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Prepare boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, w, h])
            labels.append(ann["category_id"])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply transforms
        if self.transform:
            orig_w, orig_h = image.size
            image = self.transform(image)
            new_h, new_w = image.shape[1:]  # Transformed image dimensions
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, 0] *= scale_x  # x
            boxes[:, 1] *= scale_y  # y
            boxes[:, 2] *= scale_x  # width
            boxes[:, 3] *= scale_y  # height

        # Apply Mosaic and MixUp augmentations
        if self.mosaic and np.random.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            mosaic_ids = sample(self.image_ids, 3)
            mosaic_ids.append(image_id)
            shuffle(mosaic_ids)
             
            image, boxes, labels = self.get_random_data_with_mosaic(mosaic_ids, self.input_shape)
            
            if self.mixup and np.random.rand() < self.mixup_prob:
                mixup_id = sample(self.image_ids, 1)[0]
                mixup_image, mixup_boxes, mixup_labels = self.get_random_data(mixup_id, self.input_shape, random=self.train)
                image, boxes, labels = self.get_random_data_with_mixup(image, boxes, labels, mixup_image, mixup_boxes, mixup_labels)

        else:
            image, boxes, labels = self.get_random_data(image_id, self.input_shape, random=self.train)

        # Prepare target dictionary
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        # Convert image to tensor and normalize
        image = np.array(image, dtype=np.float32)
        image = np.transpose(preprocess_input(image), (2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)

        return image, target

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image_id, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4, random=True):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        iw, ih = image.size
        h, w = input_shape

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, bw, bh, ann["category_id"]])
        boxes = np.array(boxes, dtype=np.float32)

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(boxes) > 0:
                boxes[:, 0] = boxes[:, 0] * scale + dx
                boxes[:, 1] = boxes[:, 1] * scale + dy
                boxes[:, 2] = boxes[:, 2] * scale
                boxes[:, 3] = boxes[:, 3] * scale

                boxes[:, 0] = np.maximum(boxes[:, 0], 0)
                boxes[:, 1] = np.maximum(boxes[:, 1], 0)
                boxes[:, 0] = np.minimum(boxes[:, 0], w - boxes[:, 2])
                boxes[:, 1] = np.minimum(boxes[:, 1], h - boxes[:, 3])

                valid_mask = np.logical_and(boxes[:, 2] > 1, boxes[:, 3] > 1)
                boxes = boxes[valid_mask]

            return image_data, boxes[:, :4], boxes[:, 4].astype(np.int64)

        # Random augmentation
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        
        # Flip
        if self.rand() < 0.5:
            new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes[:, 0] = w - (boxes[:, 0] + boxes[:, 2])

        # HSV augmentation
        image_data = np.array(new_image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        lut_hue = ((np.arange(0, 256, dtype=np.float32) * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(np.arange(0, 256, dtype=np.float32) * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(np.arange(0, 256, dtype=np.float32) * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # Adjust boxes
        boxes[:, 0] = boxes[:, 0] * nw / iw + dx
        boxes[:, 1] = boxes[:, 1] * nh / ih + dy
        boxes[:, 2] = boxes[:, 2] * nw / iw
        boxes[:, 3] = boxes[:, 3] * nh / ih

        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        boxes[:, 0] = np.minimum(boxes[:, 0], w - boxes[:, 2])
        boxes[:, 1] = np.minimum(boxes[:, 1], h - boxes[:, 3])

        valid_mask = np.logical_and(boxes[:, 2] > 1, boxes[:, 3] > 1)
        boxes = boxes[valid_mask]

        return image_data, boxes[:, :4], boxes[:, 4].astype(np.int64)

    def get_random_data_with_mosaic(self, image_ids, input_shape):
        h, w = input_shape
        min_offset = 0.2
        cutx = np.random.randint(int(w * min_offset), int(w * (1 - min_offset)))
        cuty = np.random.randint(int(h * min_offset), int(h * (1 - min_offset)))

        image_datas = []
        box_datas = []
        label_datas = []
        for i, image_id in enumerate(image_ids):
            img, boxes, labels = self.get_random_data(image_id, input_shape, random=True)
            image_datas.append(img)
            box_datas.append(boxes)
            label_datas.append(labels)

        # Create mosaic image
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_image[:cuty, :cutx] = image_datas[0][:cuty, :cutx]
        new_image[:cuty, cutx:] = image_datas[1][:cuty, cutx:]
        new_image[cuty:, :cutx] = image_datas[2][cuty:, :cutx]
        new_image[cuty:, cutx:] = image_datas[3][cuty:, cutx:]

        # Combine boxes
        new_boxes = []
        new_labels = []
        for i in range(4):
            if len(box_datas[i]) == 0:
                continue
            boxes = box_datas[i].copy()
            labels = label_datas[i].copy()
             
            if i == 0:  # Top-left
                boxes[:, 2] = np.minimum(boxes[:, 2], cutx - boxes[:, 0])
                boxes[:, 3] = np.minimum(boxes[:, 3], cuty - boxes[:, 1])
            elif i == 1:  # Top-right
                boxes[:, 0] = np.maximum(boxes[:, 0], cutx)
                boxes[:, 2] = np.minimum(boxes[:, 2], w - boxes[:, 0])
                boxes[:, 3] = np.minimum(boxes[:, 3], cuty - boxes[:, 1])
            elif i == 2:  # Bottom-left
                boxes[:, 1] = np.maximum(boxes[:, 1], cuty)
                boxes[:, 2] = np.minimum(boxes[:, 2], cutx - boxes[:, 0])
                boxes[:, 3] = np.minimum(boxes[:, 3], h - boxes[:, 1])
            elif i == 3:  # Bottom-right
                boxes[:, 0] = np.maximum(boxes[:, 0], cutx)
                boxes[:, 1] = np.maximum(boxes[:, 1], cuty)
                boxes[:, 2] = np.minimum(boxes[:, 2], w - boxes[:, 0])
                boxes[:, 3] = np.minimum(boxes[:, 3], h - boxes[:, 1])

            valid_mask = np.logical_and(boxes[:, 2] > 1, boxes[:, 3] > 1)
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]
            
            if len(boxes) > 0:
                new_boxes.append(boxes)
                new_labels.append(labels)

        if len(new_boxes) == 0:
            return new_image, np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
        new_boxes = np.concatenate(new_boxes, axis=0)
        new_labels = np.concatenate(new_labels, axis=0)
        return new_image, new_boxes, new_labels

    def get_random_data_with_mixup(self, image1, boxes1, labels1, image2, boxes2, labels2):
        # Create blended image
        new_image = (np.array(image1, np.float32) * 0.5 + 
                    np.array(image2, np.float32) * 0.5).astype(np.uint8)
        
        # Ensure boxes are 2D even if empty
        boxes1 = boxes1.reshape(-1, 4) if boxes1.size else np.empty((0, 4), dtype=np.float32)
        boxes2 = boxes2.reshape(-1, 4) if boxes2.size else np.empty((0, 4), dtype=np.float32)
        
        # Combine annotations
        new_boxes = np.concatenate([boxes1, boxes2], axis=0)
        new_labels = np.concatenate([labels1, labels2], axis=0)
        
        return new_image, new_boxes, new_labels