# 基于CT的松材线虫病检测系统

> **东北林业大学 计算机与控制工程学院 毕业论文项目**
> 
> **学生**: 刘昕月 (2022222997) | **指导教师**: 邱兆文 教授

---

## 项目简介

本项目开发了一个基于 CT 影像的松材线虫病可视化检测系统，通过深度学习技术实现对松材 CT 图像的自动化分析。

### 核心功能

- 🏥 **CT 图像处理**: 支持图像去噪声、增强等预处理
- 🎯 **多模型检测**: 集成 YOLOv8、U-Net、Mask R-CNN 等多种深度学习模型
- 🌐 **Web 可视化界面**: Vue.js 前端 + Flask 后端的完整 Web 应用
- 📊 **实时检测**: 支持图像上传、实时检测与结果可视化

### 技术架构

| 层级 | 技术栈 |
|------|--------|
| 前端 | Vue 3 + Vite + Element Plus |
| 后端 | Flask + Python |
| 深度学习 | PyTorch + Ultralytics |
| 图像处理 | OpenCV |

---

## 快速开始

### 1. 环境要求

- Python 3.8+
- Node.js 16+ (用于前端)
- 4GB+ RAM (推荐 8GB)
- GPU (可选，用于加速训练)

### 2. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/liumengyue1234/biye.git
cd biye

# 安装后端依赖
cd backend
pip install -r requirements.txt

# 安装前端依赖 (需要 Node.js)
cd ../frontend
npm install
```

### 3. 训练模型

```bash
# 训练 YOLOv8 (推荐 - 速度最快)
python train_yolov8.py

# 训练 U-Net
python train_unet.py

# 训练 Mask R-CNN
python train_maskrcnn.py
```

训练完成的模型权重保存在 `weights/` 目录。

### 4. 启动服务

**方式一: 一键启动**
```bash
python start_system.py
```

**方式二: 手动启动**

1. 启动后端:
```bash
cd backend
python app.py
```

2. 启动前端 (新终端):
```bash
cd frontend
npm run dev
```

### 5. 使用系统

1. 打开浏览器访问 `http://localhost:3000`
2. 上传 CT 图像或粘贴 Base64 编码
3. 选择检测模型 (YOLOv8 / U-Net / Mask R-CNN)
4. 调整参数并点击"开始检测"
5. 查看检测结果并下载

---

## 项目结构

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
├── models/                    # 模型定义
├── dataset_yolo/              # YOLO 格式数据集
├── weights/                   # 训练好的模型权重
├── results/                   # 检测结果
├── train_yolov8.py           # YOLOv8 训练脚本
├── train_unet.py             # U-Net 训练脚本
├── train_maskrcnn.py         # Mask R-CNN 训练脚本
├── convert_labelme_to_yolo.py # 数据集转换工具
├── evaluate_models.py        # 模型评估脚本
├── start_system.py           # 一键启动脚本
└── README.md                 # 本文件
```

---

## API 接口

### 健康检查
```bash
GET /api/health
```

### 图像去噪
```bash
POST /api/denoise
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "method": "bilateral|gaussian|median"
}
```

### YOLOv8 检测
```bash
POST /api/detect/yolov8
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "confidence": 0.5
}
```

### U-Net 分割
```bash
POST /api/detect/unet
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

### Mask R-CNN 分割
```bash
POST /api/detect/maskrcnn
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

### 获取可用模型
```bash
GET /api/models
```

---

## 数据集说明

### 数据来源
- **first.zip**: 567 张带标注的 CT 图像
- **third.zip**: 870 张带标注的 CT 图像

### 数据格式
- **原始格式**: Labelme JSON (矩形框标注)
- **转换格式**: YOLO TXT (class_id x_center y_center width height)

### 数据划分
| 数据集 | 数量 | 用途 |
|--------|------|------|
| 训练集 | ~567 | 模型训练 |
| 验证集 | ~870 | 模型评估 |

### 标注类别
| 类别 ID | 名称 | 描述 |
|---------|------|------|
| 0 | Vector_insect | 松材线虫 (Vector insect) |

---

## 模型说明

### 1. YOLOv8 (推荐)

| 属性 | 值 |
|------|-----|
| 类型 | 目标检测 |
| 预训练模型 | yolov8n.pt |
| 输入尺寸 | 512×512 |
| 参数量 | 3.2M |
| 特点 | 速度快，适合实时应用 |

### 2. U-Net

| 属性 | 值 |
|------|-----|
| 类型 | 语义分割 |
| 骨干网络 | ResNet 风格编码器 |
| 输入尺寸 | 512×512 |
| 参数量 | ~31M |
| 特点 | 医学图像分割经典架构 |

### 3. Mask R-CNN

| 属性 | 值 |
|------|-----|
| 类型 | 实例分割 |
| 骨干网络 | ResNet-50 + FPN |
| 输入尺寸 | 任意尺寸 |
| 参数量 | ~63M |
| 特点 | 可区分重叠目标实例 |

---

## 评估指标

### 目标检测
- **mAP@50**: IoU 阈值为 0.5 时的平均精度
- **mAP@50-95**: IoU 阈值为 0.5-0.95 时的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

### 图像分割
- **Dice**: Dice 系数 (2×交集/并集)
- **IoU**: 交并比

---

## 开发指南

### 添加新模型

1. 在 `models/` 目录创建模型文件
2. 创建对应的训练脚本
3. 在 `backend/app.py` 添加新的 API 端点
4. 在 `frontend/src/App.vue` 添加模型选择选项

### 自定义数据集

1. 收集 CT 图像
2. 使用 [Labelme](https://github.com/wkentaro/labelme) 进行标注
3. 运行 `convert_labelme_to_yolo.py` 转换为 YOLO 格式
4. 更新 `dataset_yolo/pine_nematode.yaml` 配置
5. 重新训练模型

---

## 常见问题

### Q: 模型训练太慢怎么办？

A: 
- 使用 GPU 加速 (推荐 NVIDIA CUDA)
- 减少训练轮数 (epochs)
- 使用更小的输入尺寸

### Q: 内存不足怎么办？

A:
- 减小 batch_size
- 使用更轻量的模型 (yolov8n)

### Q: 前端无法连接后端？

A:
- 确保后端服务已启动 (端口 5000)
- 检查 CORS 配置

---

## 项目信息

| 项目 | 内容 |
|------|------|
| 学校 | 东北林业大学 |
| 学院 | 计算机与控制工程学院 |
| 专业 | 软件工程 2022 级 4 班 |
| 学号 | 2022222997 |
| 学生 | 刘昕月 |
| 指导教师 | 邱兆文 教授 |
| 企业导师 | 高启 工程师 |
| 完成日期 | 2025 年 11 月 |

---

## 论文要求

- [x] 论文字数: 10000 字以上
- [ ] 外文文献: 10 篇以上
- [x] 系统代码: 完整前后端实现
- [x] 深度学习模型: U-Net, Mask R-CNN, YOLOv8
- [ ] 模型训练与评估

---

## 致谢

感谢东北林业大学计算机与控制工程学院的各位老师和同学在项目开发过程中的指导和帮助。

特别感谢:
- 邱兆文 教授 (指导教师)
- 高启 工程师 (企业导师)

---

## License

本项目仅用于学术研究和毕业论文答辩目的。
