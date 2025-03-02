import copy
import math

import torch
import torch.nn as nn

from utils.tal import dist2bbox, make_anchors
from model.nn.conv import Conv, DWConv
from model.nn.block import DFL

class Detect(nn.Module):
    """YOLO Detect head for detection models."""
    
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end mode
    max_det = 300  # max detections
    shape = None
    anchors = torch.empty(0)  # initialize anchors
    strides = torch.empty(0)  # initialize strides
    legacy = False  # backward compatibility

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with dynamic stride computation."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides to be computed dynamically

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

        # Compute strides dynamically
        self.build_strides()

    def build_strides(self):
        """Dynamically computes stride values by performing a dummy forward pass."""
        s = 256  # Assume minimum stride 2x
        dummy_input = [torch.zeros(1, 3, s, s)] * self.nl  # Create a list of dummy inputs
        with torch.no_grad():
            output = self.forward(dummy_input)  # Perform forward pass
        self.stride = torch.tensor([s / x.shape[-2] for x in output])  # Compute strides
        self.strides = self.stride  # Store computed strides

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, requires computed strides."""
        m = self  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, self.stride):  
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls bias adjustment
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, self.stride):  
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls bias adjustment

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)
