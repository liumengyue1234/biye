#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Labelme 标注格式转换为 YOLO 格式
Labelme JSON → YOLO txt (class_id, x_center, y_center, width, height) 归一化
"""

import os
import json
import shutil
from pathlib import Path

# 类别映射（根据任务书，标注为 "Vector insect" 或中文标签）
LABEL_MAP = {
    "Vector insect": 0,
    "松材线虫": 0,
    "病変区域": 0,
    "lesion": 0,
    "nematode": 0,
}


def convert_labelme_to_yolo(json_path, output_txt_path, img_width, img_height):
    """将单个 Labelme JSON 文件转换为 YOLO txt 格式"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []
    shapes = data.get('shapes', [])

    for shape in shapes:
        label = shape.get('label', '')
        if label not in LABEL_MAP:
            # 如果是未知标签，跳过或默认为0
            class_id = 0
        else:
            class_id = LABEL_MAP[label]

        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'rectangle')

        if shape_type == 'rectangle' and len(points) == 2:
            # Rectangle: [x1, y1], [x2, y2]
            x1, y1 = points[0]
            x2, y2 = points[1]
        elif shape_type == 'polygon':
            # Polygon: use bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            continue

        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_width))
        x2 = max(0, min(x2, img_width))
        y1 = max(0, min(y1, img_height))
        y2 = max(0, min(y2, img_height))

        # 转换为 YOLO 格式 (中心点 + 宽高，归一化)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        w = abs(x2 - x1) / img_width
        h = abs(y2 - y1) / img_height

        # 过滤无效框
        if w <= 0 or h <= 0:
            continue

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def process_dataset(dataset_dir, output_dir, copy_images=True):
    """
    处理整个数据集：
    1. 找到所有 JSON 文件
    2. 转换为 YOLO 格式
    3. 复制/移动图像到输出目录
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有 JSON 文件
    json_files = list(dataset_path.rglob('*.json'))
    print(f"找到 {len(json_files)} 个 JSON 标注文件")

    processed = 0
    for json_file in json_files:
        # 读取 JSON 获取图像尺寸和图像路径
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"跳过无效 JSON: {json_file}, 错误: {e}")
            continue

        img_width = data.get('imageWidth', 512)
        img_height = data.get('imageHeight', 512)
        img_filename = data.get('imagePath', '')

        # 找到对应的图像文件
        if not img_filename:
            # 尝试用相同文件名但不同扩展名
            img_stem = json_file.stem
            possible_imgs = list(json_file.parent.glob(f"{img_stem}.*"))
            img_file = None
            for p in possible_imgs:
                if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    img_file = p
                    break
        else:
            img_file = json_file.parent / img_filename
            if not img_file.exists():
                # 尝试在当前目录查找
                img_stem = Path(img_filename).stem
                possible_imgs = list(json_file.parent.glob(f"{img_stem}.*"))
                img_file = None
                for p in possible_imgs:
                    if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                        img_file = p
                        break

        if img_file is None or not img_file.exists():
            print(f"找不到图像文件: {json_file}")
            continue

        # 输出路径
        txt_filename = img_file.stem + '.txt'
        output_txt = output_path / txt_filename
        output_img = output_path / img_file.name

        # 转换标注
        convert_labelme_to_yolo(str(json_file), str(output_txt), img_width, img_height)

        # 复制图像
        if copy_images:
            shutil.copy(str(img_file), str(output_img))

        processed += 1
        if processed % 50 == 0:
            print(f"已处理 {processed}/{len(json_files)} 个文件")

    print(f"完成！共处理 {processed} 个文件")
    return processed


if __name__ == '__main__':
    base_dir = 'C:/Users/86157/WorkBuddy/2026-05-12-task-2'

    print("=" * 50)
    print("开始转换 first 数据集...")
    print("=" * 50)
    n1 = process_dataset(
        os.path.join(base_dir, 'dataset_first'),
        os.path.join(base_dir, 'dataset_yolo', 'train'),
        copy_images=True
    )

    print("=" * 50)
    print("开始转换 third 数据集...")
    print("=" * 50)
    n2 = process_dataset(
        os.path.join(base_dir, 'dataset_third'),
        os.path.join(base_dir, 'dataset_yolo', 'val'),
        copy_images=True
    )

    print("=" * 50)
    print(f"总计处理: first={n1}, third={n2}")
    print("=" * 50)
