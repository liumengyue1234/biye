# 基于CT的松材线虫病检测系统 - 使用指南

## 📋 系统概述

本系统是基于 CT 影像的松材线虫病检测系统，集成了多种深度学习模型，支持图像去噪、目标检测和图像分割。

### 核心功能

- 🔬 **CT图像去噪**: 双边滤波、高斯滤波、中值滤波
- 🎯 **目标检测**: YOLOv8 模型
- 📊 **图像分割**: U-Net、Mask R-CNN 模型
- 🌐 **Web界面**: Vue.js 前端 + Flask 后端

---

## 🚀 快速开始

### 方法一：一键启动（推荐）

```bash
cd C:/Users/86157/WorkBuddy/2026-05-12-task-2
python start_system.py
```

然后在浏览器中打开 `http://localhost:3000`

### 方法二：手动启动

#### 1. 启动后端服务

```bash
# 进入后端目录
cd backend

# 安装依赖（如果尚未安装）
pip install -r requirements.txt

# 启动服务
python app.py
```

后端服务将在 `http://localhost:5000` 运行

#### 2. 启动前端服务（新终端）

```bash
# 进入前端目录
cd frontend

# 安装依赖（如果尚未安装）
npm install

# 启动开发服务器
npm run dev
```

前端服务将在 `http://localhost:3000` 运行

---

## 📖 使用说明

### 1. 上传图像

有两种方式上传 CT 图像：

**方式一：拖拽上传**
- 将图像文件拖拽到上传区域
- 支持 PNG、JPG、BMP 格式

**方式二：Base64 粘贴**
- 复制图像的 Base64 编码
- 粘贴到文本框中
- 点击"加载图像"按钮

### 2. 选择模型

系统支持三种检测模型：

| 模型 | 适用场景 | 特点 |
|------|----------|------|
| YOLOv8 | 目标检测 | 速度快，精度高 |
| U-Net | 语义分割 | 适合医学图像 |
| Mask R-CNN | 实例分割 | 可区分重叠目标 |

### 3. 调整参数

**置信度阈值**：
- 范围：0.00 - 1.00
- 默认值：0.50
- 较低的值会检测更多目标，但可能有更多误检

**去噪方法**：
- 双边滤波（推荐）：保边去噪
- 高斯滤波：平滑处理
- 中值滤波：去除椒盐噪声

### 4. 运行检测

1. 点击"开始检测"按钮
2. 等待处理完成
3. 查看检测结果
4. 可点击"下载结果"保存带标注的图像

---

## 🔧 API 接口使用

### Python 调用示例

```python
import requests
import base64

# 读取图像
with open('ct_image.png', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')

# 调用检测接口
response = requests.post('http://localhost:5000/api/detect/yolov8', json={
    'image': img_data,
    'confidence': 0.5
})

result = response.json()
print(f"检测到 {result['count']} 个目标")
print(result['detections'])
```

### cURL 调用示例

```bash
# 健康检查
curl http://localhost:5000/api/health

# 图像去噪
curl -X POST http://localhost:5000/api/denoise \
  -H "Content-Type: application/json" \
  -d '{"image": "base64...", "method": "bilateral"}'

# YOLOv8 检测
curl -X POST http://localhost:5000/api/detect/yolov8 \
  -H "Content-Type: application/json" \
  -d '{"image": "base64...", "confidence": 0.5}'
```

---

## 🏋️ 模型训练

### 训练 YOLOv8

```bash
python train_yolov8.py
```

参数说明：
- `epochs`: 训练轮数（默认50）
- `imgsz`: 输入图像尺寸（默认512）
- `batch`: 批次大小（默认8）

### 训练 U-Net

```bash
python train_unet.py
```

### 训练 Mask R-CNN

```bash
python train_maskrcnn.py
```

### 模型评估

```bash
python evaluate_models.py
```

---

## 📁 项目结构

```
biye/
├── backend/                    # Flask 后端
│   ├── app.py                  # API 服务主文件
│   └── requirements.txt        # Python 依赖
├── frontend/                   # Vue 3 前端
│   ├── src/
│   │   ├── App.vue            # 主组件
│   │   └── main.js            # 入口文件
│   └── package.json           # Node 依赖
├── dataset_yolo/              # YOLO 格式数据集
├── weights/                   # 训练好的模型权重
│   ├── yolov8n_best.pt       # YOLOv8 模型
│   ├── unet_best.pth          # U-Net 模型
│   └── maskrcnn_final.pth     # Mask R-CNN 模型
├── train_yolov8.py           # YOLOv8 训练脚本
├── train_unet.py             # U-Net 训练脚本
├── train_maskrcnn.py         # Mask R-CNN 训练脚本
├── evaluate_models.py        # 模型评估脚本
└── start_system.py          # 一键启动脚本
```

---

## ❓ 常见问题

### Q1: 后端启动失败？

**可能原因**：
- 端口 5000 被占用
- 依赖未安装

**解决方案**：
```bash
# 检查端口占用
netstat -ano | findstr 5000

# 安装依赖
pip install -r backend/requirements.txt
```

### Q2: 前端无法连接后端？

**可能原因**：
- 后端未启动
- CORS 跨域问题

**解决方案**：
1. 确保后端运行在 `http://localhost:5000`
2. 检查浏览器控制台错误信息

### Q3: 模型检测效果不好？

**可能原因**：
- 训练数据不足
- 图像质量差
- 置信度设置不当

**解决方案**：
1. 收集更多标注数据
2. 调整置信度阈值
3. 尝试不同的模型

### Q4: 内存不足？

**解决方案**：
- 减小 batch_size
- 使用更小的图像尺寸
- 关闭其他占用内存的程序

---

## 📞 获取帮助

如有其他问题，请联系：
- **学生**: 刘昕月
- **学号**: 2022222997
- **学校**: 东北林业大学 计算机与控制工程学院
- **指导教师**: 邱兆文 教授

---

## 📄 许可证

本项目仅用于学术研究和毕业论文答辩目的。
