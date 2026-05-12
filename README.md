# 基于CT的松材线虫病检测系统

## 项目概述

本项目是东北林业大学计算机与控制工程学院的毕业论文项目，旨在开发一个基于CT影像的松材线虫病可视化检测系统。

### 功能特点

- 🔬 **CT图像处理**：支持CT图像去噪声处理
- 🎯 **多种检测模型**：集成 U-Net、Mask R-CNN、YOLOv8 等深度学习模型
- 🌐 **Web界面**：提供直观的 Vue.js 前端界面
- 📊 **实时检测**：支持图像上传、实时检测和结果可视化

### 技术栈

**前端**：
- Vue.js 3 + Vite
- Element Plus UI 组件库
- Axios HTTP 客户端

**后端**：
- Flask (Python)
- PyTorch 深度学习框架
- OpenCV 图像处理

**深度学习模型**：
- U-Net：医学图像分割
- Mask R-CNN：实例分割
- YOLOv8：目标检测

---

## 项目结构

```
.
├── dataset_first/              # 数据集1 (标注数据)
├── dataset_third/              # 数据集2 (标注数据)
├── dataset_yolo/               # YOLO格式数据集
│   ├── train/                  # 训练集
│   └── val/                    # 验证集
├── models/                     # 模型定义
│   ├── unet/                   # U-Net模型
│   ├── maskrcnn/               # Mask R-CNN模型
│   └── yolov8/                 # YOLOv8模型
├── backend/                    # Flask后端
│   ├── app.py                  # 主应用
│   └── requirements.txt        # 依赖
├── frontend/                   # Vue前端
│   ├── src/                    # 源代码
│   └── package.json            # 依赖
├── weights/                    # 训练好的模型权重
├── results/                    # 检测结果
├── convert_labelme_to_yolo.py  # 数据集转换脚本
├── train_yolov8.py             # YOLOv8训练脚本
├── train_unet.py               # U-Net训练脚本
└── train_maskrcnn.py           # Mask R-CNN训练脚本
```

---

## 快速开始

### 1. 环境配置

**Python 环境要求**：
- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU训练)

**安装依赖**：

```bash
# 安装后端依赖
cd backend
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
```

### 2. 数据集准备

项目已包含预处理的数据集，格式为 YOLO TXT 格式。

如需重新转换数据集：

```bash
python convert_labelme_to_yolo.py
```

### 3. 模型训练

**训练 YOLOv8**：
```bash
python train_yolov8.py
```

**训练 U-Net**：
```bash
python train_unet.py
```

**训练 Mask R-CNN**：
```bash
python train_maskrcnn.py
```

训练完成的模型权重保存在 `weights/` 目录下。

### 4. 启动服务

**启动后端**：
```bash
cd backend
python app.py
```

后端服务运行在 `http://localhost:5000`

**启动前端**：
```bash
cd frontend
npm run dev
```

前端服务运行在 `http://localhost:3000`

### 5. 使用系统

1. 打开浏览器访问 `http://localhost:3000`
2. 上传 CT 图像（或粘贴 Base64 编码）
3. 选择检测模型（YOLOv8 / U-Net / Mask R-CNN）
4. 调整置信度阈值和去噪参数
5. 点击"开始检测"查看结果

---

## API 接口

### 健康检查
```
GET /api/health
```

### 图像去噪
```
POST /api/denoise
{
  "image": "base64_encoded_image",
  "method": "bilateral|gaussian|median"
}
```

### YOLOv8 检测
```
POST /api/detect/yolov8
{
  "image": "base64_encoded_image",
  "confidence": 0.5
}
```

### U-Net 分割
```
POST /api/detect/unet
{
  "image": "base64_encoded_image"
}
```

### Mask R-CNN 分割
```
POST /api/detect/maskrcnn
{
  "image": "base64_encoded_image"
}
```

### 获取模型列表
```
GET /api/models
```

---

## 数据集说明

### 数据来源
- `first.zip`：第一批标注数据（567张图像）
- `third.zip`：第三批标注数据（870张图像）

### 标注格式
- 原始格式：Labelme JSON
- 转换格式：YOLO TXT (class_id, x_center, y_center, width, height)

### 标注类别
- `Vector_insect`：松材线虫（Vector insect）

### 数据划分
- 训练集：first.zip 数据
- 验证集：third.zip 数据

---

## 模型说明

### YOLOv8
- 类型：目标检测
- 预训练模型：yolov8n.pt (nano版本)
- 输入尺寸：512x512
- 特点：速度快，适合实时检测

### U-Net
- 类型：语义分割
- 编码器：ResNet风格
- 输入尺寸：512x512
- 特点：医学图像分割经典架构

### Mask R-CNN
- 类型：实例分割
- 骨干网络：ResNet-50 + FPN
- 输入尺寸：任意尺寸
- 特点：可区分重叠目标实例

---

## 性能指标

| 模型 | mAP@50 | mAP@50-95 | 参数量 |
|------|--------|-----------|--------|
| YOLOv8n | - | - | 3.2M |
| U-Net | - | - | 31M |
| Mask R-CNN | - | - | 63M |

*注：需完成训练后查看实际指标*

---

## 项目信息

- **学生**：刘昕月
- **学号**：2022222997
- **专业**：软件工程2022级4班
- **指导教师**：邱兆文 教授
- **企业导师**：高启 工程师
- **学院**：计算机与控制工程学院
- **学校**：东北林业大学

---

## 论文要求

- 论文字数：10000字以上
- 外文文献：10篇以上
- 完成日期：2025年11月

---

## 致谢

感谢东北林业大学计算机与控制工程学院的各位老师和同学在项目开发过程中的指导和帮助。
