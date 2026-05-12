#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask R-CNN 松材线虫病实例分割模型训练脚本
使用 PyTorch torchvision 实现
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json


# ===================== COCO 格式数据集 =====================
class PineNematodeCOCODataset(Dataset):
    """松材线虫病 COCO 格式数据集"""
    
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        # 获取所有图像文件
        self.image_files = list(self.images_dir.glob('*.png')) + \
                          list(self.images_dir.glob('*.jpg'))
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # 读取图像
        image = Image.open(str(img_path)).convert('RGB')
        orig_w, orig_h = image.size
        image = self.transform(image)
        
        # 读取标注
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
        boxes = []
        labels = []
        masks = []
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                h, w = orig_h, orig_w
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_center, y_center, box_w, box_h = map(float, parts)
                        
                        # 转换为像素坐标
                        x1 = int((x_center - box_w / 2) * w)
                        y1 = int((y_center - box_h / 2) * h)
                        x2 = int((x_center + box_w / 2) * w)
                        y2 = int((y_center + box_h / 2) * h)
                        
                        # 确保坐标有效
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(1)  # 类别 1 (Vector insect)
                            
                            # 创建简单的掩码
                            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                            mask[y1:y2, x1:x2] = 1
                            masks.append(mask)
            except Exception as e:
                print(f"读取标注文件失败: {label_path}, 错误: {e}")
        
        # 如果没有标注，返回空目标
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
            masks = [np.zeros((orig_h, orig_w), dtype=np.uint8)]
        
        # 转换为 Tensor 格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 合并所有掩码为多通道
        if len(masks) > 0:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((1, orig_h, orig_w), dtype=torch.uint8)
        
        # 构建目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx])
        }
        
        return image, target


def collate_fn(batch):
    """自定义批量整理函数"""
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    """创建 Mask R-CNN 模型"""
    # 加载预训练的 Mask R-CNN 模型
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 替换预训练头
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 获取掩码预测器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    # 替换掩码头
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


def train_mask_rcnn():
    """训练 Mask R-CNN 模型"""
    print("=" * 60)
    print("开始训练 Mask R-CNN 模型...")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    base_dir = "C:/Users/86157/WorkBuddy/2026-05-12-task-2"
    train_images = os.path.join(base_dir, "dataset_yolo/train")
    train_labels = os.path.join(base_dir, "dataset_yolo/train")
    
    # 创建数据集
    train_dataset = PineNematodeCOCODataset(train_images, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 创建模型 (类别数: 背景 + Vector insect = 2)
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 学习率调度
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 训练循环
    num_epochs = 10  # Mask R-CNN 训练轮数
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f}")
        
        epoch_loss /= len(train_loader)
        lr_scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss:.4f}")
        
        # 每5轮保存一次模型
        if (epoch + 1) % 5 == 0:
            weights_dir = os.path.join(base_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            save_path = os.path.join(weights_dir, f"maskrcnn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"已保存模型到: {save_path}")
    
    # 保存最终模型
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, "maskrcnn_final.pth")
    torch.save(model.state_dict(), save_path)
    
    print("=" * 60)
    print("Mask R-CNN 训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    train_mask_rcnn()
