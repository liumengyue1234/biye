#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net 松材线虫病分割模型训练脚本
用于 CT 图像的病变区域分割
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


# ===================== U-Net 模型定义 =====================
class DoubleConv(nn.Module):
    """双层卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 分割模型"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 编码器（下采样）
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # 解码器（上采样）
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # 输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        return torch.sigmoid(self.final_conv(x))


# ===================== 数据集类 =====================
class PineNematodeDataset(Dataset):
    """松材线虫病 CT 图像数据集"""
    
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        # 获取所有图像文件
        self.image_files = list(self.images_dir.glob('*.png')) + \
                          list(self.images_dir.glob('*.jpg'))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # 读取图像
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((512, 512), dtype=np.uint8)
        else:
            image = cv2.resize(image, (512, 512))
        
        # 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 读取标注掩码
        label_path = self.labels_dir / (img_path.stem + '.txt')
        mask = np.zeros((512, 512), dtype=np.float32)
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                h, w = image.shape[:2]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO 格式: class x_center y_center width height
                        _, x_center, y_center, box_w, box_h = map(float, parts)
                        
                        # 转换为像素坐标
                        x1 = int((x_center - box_w / 2) * w)
                        y1 = int((y_center - box_h / 2) * h)
                        x2 = int((x_center + box_w / 2) * w)
                        y2 = int((y_center + box_h / 2) * h)
                        
                        # 确保坐标在有效范围内
                        x1, x2 = max(0, x1), min(w, x2)
                        y1, y2 = max(0, y1), min(h, y2)
                        
                        # 绘制矩形掩码
                        if x2 > x1 and y2 > y1:
                            mask[y1:y2, x1:x2] = 1.0
            except Exception as e:
                print(f"读取标注文件失败: {label_path}, 错误: {e}")
        
        # 转换为 PyTorch 格式 [C, H, W]
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask


# ===================== 训练函数 =====================
def train_unet():
    """训练 U-Net 模型"""
    print("=" * 60)
    print("开始训练 U-Net 模型...")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    base_dir = "C:/Users/86157/WorkBuddy/2026-05-12-task-2"
    train_images = os.path.join(base_dir, "dataset_yolo/train")
    train_labels = os.path.join(base_dir, "dataset_yolo/train")
    val_images = os.path.join(base_dir, "dataset_yolo/val")
    val_labels = os.path.join(base_dir, "dataset_yolo/val")
    
    # 创建数据集
    train_dataset = PineNematodeDataset(train_images, train_labels)
    val_dataset = PineNematodeDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # 损失函数 - 使用 Dice Loss + BCE
    criterion = nn.BCELoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练循环
    num_epochs = 30
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            weights_dir = os.path.join(base_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            save_path = os.path.join(weights_dir, "unet_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"已保存最佳模型到: {save_path}")
    
    print("=" * 60)
    print("U-Net 训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    train_unet()
