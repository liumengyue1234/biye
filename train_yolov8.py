#!/usr/bin/env python3
# -_- coding: utf-8 -_-
"""
YOLOv8 松材线虫病检测模型训练脚本
使用 Ultralytics YOLOv8 进行训练
"""

import os
import sys
from ultralytics import YOLO

def main():
    # 数据集配置文件路径
    data_yaml = "C:/Users/86157/WorkBuddy/2026-05-12-task-2/dataset_yolo/pine_nematode.yaml"
    
    # 使用 YOLOv8n 预训练模型（nano 版本，适合快速训练）
    model = YOLO('yolov8n.pt')
    
    print("=" * 60)
    print("开始训练 YOLOv8n 模型...")
    print("=" * 60)
    
    # 训练模型
    # epochs=50 可根据需要调整，这里先用50轮
    # imgsz=512 因为CT图像是512x512
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=512,
        batch=8,
        name='pine_nematode_yolov8n',
        project='C:/Users/86157/WorkBuddy/2026-05-12-task-2/runs',
        exist_ok=True,
        device='cpu',  # CPU训练，如有GPU可改为 '0'
        workers=0,
        verbose=True
    )
    
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 保存模型到 weights 目录
    weights_dir = "C:/Users/86157/WorkBuddy/2026-05-12-task-2/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # 复制最佳模型
    best_model = "C:/Users/86157/WorkBuddy/2026-05-12-task-2/runs/pine_nematode_yolov8n/weights/best.pt"
    target_path = os.path.join(weights_dir, "yolov8n_best.pt")
    
    if os.path.exists(best_model):
        import shutil
        shutil.copy(best_model, target_path)
        print(f"最佳模型已保存到: {target_path}")
    else:
        print(f"警告: 未找到训练好的模型: {best_model}")
    
    # 验证模型
    print("\n开始验证模型...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    print("\nYOLOv8 训练流程完成！")


if __name__ == '__main__':
    main()
