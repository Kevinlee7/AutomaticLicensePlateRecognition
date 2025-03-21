#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
lpr_recognizer.py

封装了LPRNet车牌识别的功能为一个类 LPRRecognizer，
在初始化时加载模型，之后可以多次调用 get_plate_text 方法对图像进行推理并返回解码后的车牌文本。

使用示例：
    from lpr_recognizer import LPRRecognizer
    import cv2

    recognizer = LPRRecognizer(model_path="./models/demo.pth", device=torch.device("cpu"))
    image = cv2.imread("test.jpg")
    plate_texts = recognizer.get_plate_text(image)
    print("识别结果：", plate_texts)
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

# --------------------- 模型定义 ---------------------

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),
            small_basic_block(ch_in=128, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)
        global_context = []
        h, w = None, None
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            if i == 0:
                h, w = f.size(2), f.size(3)
            else:
                h = min(h, f.size(2))
                w = min(w, f.size(3))
            f_mean = torch.mean(torch.pow(f, 2))
            f = f / (f_mean + 1e-6)
            global_context.append(f)
        for i in range(len(global_context)):
            if global_context[i].size(2) != h or global_context[i].size(3) != w:
                global_context[i] = nn.functional.adaptive_avg_pool2d(global_context[i], (h, w))
        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits

# 字符集及映射
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

# --------------------- 解码函数 ---------------------

def greedy_decode(preds, pred_char=False):
    """
    对模型输出进行贪婪解码：
      preds: numpy数组，形状为 (batch_size, num_classes, sequence_length)
    返回：
      如果 pred_char 为 True，则返回字符标签列表；否则返回索引列表。
    """
    last_chars_idx = len(CHARS) - 1
    pred_labels = []
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]  # shape: (num_classes, seq_len)
        pred_label = []
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = []
        pre_c = -1
        for c in pred_label:
            if (pre_c == c) or (c == last_chars_idx):
                pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
    if pred_char:
        labels = []
        for label in pred_labels:
            s = ""
            for i in label:
                s += CHARS[i]
            labels.append(s)
        return pred_labels, labels
    else:
        return pred_labels

# --------------------- LPRRecognizer 类 ---------------------

class LPRRecognizer:
    """
    车牌识别类，初始化时加载模型，之后可多次调用 get_plate_text 对图像进行推理，
    返回解码后的车牌文本列表。
    """
    def __init__(self, model_path, device=torch.device("cpu")):
        self.device = device
        self.model = self._load_model(model_path, device)

    def _load_model(self, model_path, device):
        model = LPRNet(lpr_max_len=8, class_num=len(CHARS), dropout_rate=0.5)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def _preprocess_image(self, image):
        """
        对输入图像（NumPy数组，BGR格式）进行预处理：
          1. 调整为 (94, 24) 大小；
          2. 归一化到[0, 1]；
          3. 转换通道顺序，从 (H, W, C) 到 (C, H, W)；
          4. 增加 batch 维度。
        """
        if image is None:
            raise ValueError("输入图像为空")
        height, width, _ = image.shape
        if width != 94 or height != 24:
            image = cv2.resize(image, (94, 24))
        image = image.astype('float32') / 255.0
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        image = np.expand_dims(image, axis=0)    # (1, 3, 24, 94)
        tensor_image = torch.from_numpy(image)
        return tensor_image

    def _infer(self, tensor_image):
        """
        使用模型进行推理，并返回模型输出的 NumPy 数组。
        """
        with torch.no_grad():
            output = self.model(tensor_image.to(self.device))
            output = output.cpu().detach().numpy()
        return output

    def get_plate_text(self, image):
        """
        对输入图像（NumPy数组，BGR格式）进行车牌识别，
        返回解码后的车牌文本列表。
        """
        tensor_image = self._preprocess_image(image)
        output = self._infer(tensor_image)
        _, plate_texts = greedy_decode(output, pred_char=True)
        return plate_texts

# --------------------- 测试代码（可选） ---------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LPRRecognizer Test")
    parser.add_argument("--image", type=str, default="test.jpg", help="待识别的图像文件路径")
    parser.add_argument("--model_path", type=str, default="./models/demo.pth", help="预训练模型权重路径")
    args = parser.parse_args()

    # 读取图像（BGR格式）
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"无法加载图像：{args.image}")

    recognizer = LPRRecognizer(model_path=args.model_path, device=torch.device("cpu"))
    plate_texts = recognizer.get_plate_text(image)
    print("识别结果：", plate_texts)
