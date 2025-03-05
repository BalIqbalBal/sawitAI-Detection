# reference https://github.com/bubbliiiing/yolov8-pytorch/blob/master/utils/dataloader.py

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
    def __init__(self, root_dir, annotation_file, input_shape, num_classes, epoch_length, \
                 mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7, transform=None):
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

        self.bbox_attrs = 5 + num_classes

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
        """
        Args:
            index (int): Index of the image to retrieve.

        Returns:
            image (torch.Tensor): The image as a tensor.
            target (dict): A dictionary containing annotations (bboxes in xyxy format, labels, etc.).
        """
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
            boxes.append([x, y, x + w, y + h])  # Keep in xyxy format
            labels.append(ann["category_id"])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Apply transforms (if any)
        if self.transform:
            orig_w, orig_h = image.size
            image = self.transform(image)

            new_h, new_w = image.shape[1:]  # Transformed image dimensions
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, [0, 2]] *= scale_x  # Scale x-coordinates
            boxes[:, [1, 3]] *= scale_y  # Scale y-coordinates

        # Apply Mosaic and MixUp augmentations
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            # Sample 3 additional images for Mosaic
            mosaic_ids = sample(self.image_ids, 3)
            mosaic_ids.append(image_id)
            shuffle(mosaic_ids)

            # Apply Mosaic augmentation
            image, boxes, labels = self.get_random_data_with_Mosaic(mosaic_ids, self.input_shape)

            # Apply MixUp augmentation (if enabled)
            if self.mixup and self.rand() < self.mixup_prob:
                mixup_id = sample(self.image_ids, 1)[0]
                mixup_image, mixup_boxes, mixup_labels = self.get_random_data(mixup_id, self.input_shape, random=self.train)
                image, boxes, labels = self.get_random_data_with_MixUp(image, boxes, labels, mixup_image, mixup_boxes, mixup_labels)
        else:
            # Apply standard augmentation
            image, boxes, labels = self.get_random_data(image_id, self.input_shape, random=self.train)

        # Prepare target dictionary (keep boxes in xyxy format)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),  # Boxes in xyxy format
            "labels": torch.as_tensor(labels, dtype=torch.int64),  # Class labels
        }

        # Convert image to tensor and normalize
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)

        return image, target

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image_id, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        iw, ih = image.size
        h, w = input_shape

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        box = np.array([np.array([x, y, x + w, y + h, ann["category_id"]]) for ann in anns for x, y, w, h in [ann["bbox"]]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
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
        image = new_image

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, image_ids, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for image_id in image_ids:
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.root_dir, image_info["file_name"])
            image = Image.open(image_path).convert("RGB")
            iw, ih = image.size

            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            box = np.array([np.array([x, y, x + w, y + h, ann["category_id"]]) for ann in anns for x, y, w, h in [ann["bbox"]]])

            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


# DataLoader中collate_fn使用
def yolo_collate_fn(batch):
    """
    Collate function for YoloCOCODataset.
    Args:
        batch: List of tuples (image, targets).
    Returns:
        images: Stacked tensor of shape (batch_size, C, H, W).
        targets: List of dictionaries containing boxes and labels.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, targets