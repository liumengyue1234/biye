#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
松材线虫病检测系统 - Flask 后端 API
提供图像去噪、分割、检测等接口
"""

import os
import io
import base64
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 全局变量存储模型
models = {}
MODEL_DIR = Path("C:/Users/86157/WorkBuddy/2026-05-12-task-2/weights")


# ===================== 工具函数 =====================
def base64_to_image(base64_str):
    """将 Base64 字符串转换为 OpenCV 图像"""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(img):
    """将 OpenCV 图像转换为 Base64 字符串"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def denoise_image(img, method='bilateral'):
    """
    对 CT 图像进行去噪声处理
    
    Args:
        img: OpenCV 图像 (BGR 格式)
        method: 去噪方法 ('bilateral', 'gaussian', 'median')
    
    Returns:
        去噪后的图像
    """
    if method == 'bilateral':
        # 双边滤波器 - 保边去噪
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
    elif method == 'gaussian':
        # 高斯滤波
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
    elif method == 'median':
        # 中值滤波
        denoised = cv2.medianBlur(img, 5)
    else:
        denoised = img
    
    return denoised


def load_yolov8_model():
    """加载 YOLOv8 模型"""
    try:
        from ultralytics import YOLO
        
        model_path = MODEL_DIR / "yolov8n_best.pt"
        if model_path.exists():
            model = YOLO(str(model_path))
            logger.info(f"YOLOv8 模型已加载: {model_path}")
            return model
        else:
            logger.warning(f"YOLOv8 模型文件不存在: {model_path}")
            return None
    except Exception as e:
        logger.error(f"加载 YOLOv8 模型失败: {e}")
        return None


def load_unet_model():
    """加载 U-Net 模型"""
    try:
        model_path = MODEL_DIR / "unet_best.pth"
        if model_path.exists():
            # 导入 U-Net 模型定义
            sys.path.insert(0, str(Path(__file__).parent))
            from train_unet import UNet
            
            model = UNet(in_channels=1, out_channels=1)
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            logger.info(f"U-Net 模型已加载: {model_path}")
            return model
        else:
            logger.warning(f"U-Net 模型文件不存在: {model_path}")
            return None
    except Exception as e:
        logger.error(f"加载 U-Net 模型失败: {e}")
        return None


def load_maskrcnn_model():
    """加载 Mask R-CNN 模型"""
    try:
        model_path = MODEL_DIR / "maskrcnn_final.pth"
        if model_path.exists():
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            
            model = maskrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
            
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            logger.info(f"Mask R-CNN 模型已加载: {model_path}")
            return model
        else:
            logger.warning(f"Mask R-CNN 模型文件不存在: {model_path}")
            return None
    except Exception as e:
        logger.error(f"加载 Mask R-CNN 模型失败: {e}")
        return None


# ===================== API 路由 =====================

@app.route('/')
def index():
    """首页"""
    return jsonify({
        'status': 'ok',
        'message': '松材线虫病检测系统 API 服务',
        'version': '1.0.0',
        'endpoints': [
            '/api/health - 健康检查',
            '/api/denoise - 图像去噪',
            '/api/detect/yolov8 - YOLOv8 检测',
            '/api/detect/unet - U-Net 分割',
            '/api/detect/maskrcnn - Mask R-CNN 分割',
            '/api/models - 获取可用模型列表'
        ]
    })


@app.route('/api/health')
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'service': 'pine-nematode-detection-api'
    })


@app.route('/api/denoise', methods=['POST'])
def denoise():
    """
    图像去噪接口
    
    请求参数:
        - image: Base64 编码的图像
        - method: 去噪方法 ('bilateral', 'gaussian', 'median')
    
    返回:
        - result: Base64 编码的去噪后图像
    """
    try:
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({'error': '缺少 image 参数'}), 400
        
        # 解析图像
        img = base64_to_image(data['image'])
        if img is None:
            return jsonify({'error': '无法解析图像'}), 400
        
        # 去噪处理
        method = data.get('method', 'bilateral')
        denoised = denoise_image(img, method)
        
        # 返回结果
        return jsonify({
            'success': True,
            'result': image_to_base64(denoised),
            'method': method
        })
    
    except Exception as e:
        logger.error(f"去噪处理失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/yolov8', methods=['POST'])
def detect_yolov8():
    """
    YOLOv8 目标检测接口
    
    请求参数:
        - image: Base64 编码的图像
        - confidence: 置信度阈值 (0-1)
    
    返回:
        - detections: 检测结果列表
        - result: 带标注的图像 (Base64)
    """
    try:
        global models
        
        # 加载模型（延迟加载）
        if 'yolov8' not in models:
            models['yolov8'] = load_yolov8_model()
        
        model = models['yolov8']
        if model is None:
            return jsonify({'error': 'YOLOv8 模型未加载，请先训练模型'}), 500
        
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({'error': '缺少 image 参数'}), 400
        
        # 解析图像
        img = base64_to_image(data['image'])
        if img is None:
            return jsonify({'error': '无法解析图像'}), 400
        
        # 预处理
        confidence = float(data.get('confidence', 0.5))
        
        # 执行检测
        results = model(img, conf=confidence, verbose=False)
        
        # 解析结果
        detections = []
        annotated_img = img.copy()
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class': cls,
                    'label': 'Vector_insect'
                })
                
                # 绘制边界框
                cv2.rectangle(annotated_img, 
                             (int(x1), int(y1)), 
                             (int(x2), int(y2)), 
                             (0, 255, 0), 2)
                cv2.putText(annotated_img, 
                           f'{conf:.2f}', 
                           (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 1)
        
        return jsonify({
            'success': True,
            'count': len(detections),
            'detections': detections,
            'result': image_to_base64(annotated_img)
        })
    
    except Exception as e:
        logger.error(f"YOLOv8 检测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/unet', methods=['POST'])
def detect_unet():
    """
    U-Net 分割接口
    
    请求参数:
        - image: Base64 编码的图像
    
    返回:
        - result: 分割掩码叠加图像 (Base64)
    """
    try:
        global models
        
        if 'unet' not in models:
            models['unet'] = load_unet_model()
        
        model = models['unet']
        if model is None:
            return jsonify({'error': 'U-Net 模型未加载，请先训练模型'}), 500
        
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({'error': '缺少 image 参数'}), 400
        
        # 解析图像
        img = base64_to_image(data['image'])
        if img is None:
            return jsonify({'error': '无法解析图像'}), 400
        
        # 预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (512, 512))
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为 Tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        
        # 执行分割
        with torch.no_grad():
            output = model(tensor)
        
        # 处理输出
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        # 叠加掩码到原图
        result = img.copy()
        result[mask > 0] = [0, 255, 0]  # 绿色标注
        
        return jsonify({
            'success': True,
            'mask': image_to_base64(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
            'result': image_to_base64(result)
        })
    
    except Exception as e:
        logger.error(f"U-Net 分割失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/maskrcnn', methods=['POST'])
def detect_maskrcnn():
    """
    Mask R-CNN 分割接口
    
    请求参数:
        - image: Base64 编码的图像
    
    返回:
        - detections: 检测结果
        - result: 分割结果图像 (Base64)
    """
    try:
        global models
        
        if 'maskrcnn' not in models:
            models['maskrcnn'] = load_maskrcnn_model()
        
        model = models['maskrcnn']
        if model is None:
            return jsonify({'error': 'Mask R-CNN 模型未加载，请先训练模型'}), 500
        
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({'error': '缺少 image 参数'}), 400
        
        # 解析图像
        img = base64_to_image(data['image'])
        if img is None:
            return jsonify({'error': '无法解析图像'}), 400
        
        # PIL Image 转换
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(pil_img).unsqueeze(0)
        
        # 执行检测
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # 解析结果
        pred = predictions[0]
        detections = []
        result = img.copy()
        
        for i in range(len(pred['boxes'])):
            score = pred['scores'][i].item()
            if score > 0.5:
                box = pred['boxes'][i].cpu().numpy()
                mask = pred['masks'][i, 0].cpu().numpy()
                
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': score,
                    'label': 'Vector_insect'
                })
                
                # 绘制边界框
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 叠加掩码
                mask = (mask > 0.5).astype(np.uint8)
                result[mask > 0] = [0, 255, 0]
        
        return jsonify({
            'success': True,
            'count': len(detections),
            'detections': detections,
            'result': image_to_base64(result)
        })
    
    except Exception as e:
        logger.error(f"Mask R-CNN 分割失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def list_models():
    """获取可用模型列表"""
    available_models = []
    
    # 检查各个模型
    for model_name, model_file in [
        ('YOLOv8', 'yolov8n_best.pt'),
        ('U-Net', 'unet_best.pth'),
        ('Mask R-CNN', 'maskrcnn_final.pth')
    ]:
        model_path = MODEL_DIR / model_file
        available_models.append({
            'name': model_name,
            'file': model_file,
            'available': model_path.exists(),
            'path': str(model_path)
        })
    
    return jsonify({
        'models': available_models,
        'model_dir': str(MODEL_DIR)
    })


@app.route('/api/batch', methods=['POST'])
def batch_process():
    """
    批量处理接口
    
    请求参数:
        - images: Base64 编码图像列表
        - method: 处理方法 ('denoise', 'detect')
        - model: 模型类型 ('yolov8', 'unet', 'maskrcnn')
    """
    try:
        data = request.json
        
        if not data or 'images' not in data:
            return jsonify({'error': '缺少 images 参数'}), 400
        
        images = data['images']
        method = data.get('method', 'denoise')
        model_type = data.get('model', 'yolov8')
        
        results = []
        
        for i, img_base64 in enumerate(images):
            try:
                img = base64_to_image(img_base64)
                if img is None:
                    results.append({'index': i, 'error': '无法解析图像'})
                    continue
                
                # 根据方法处理
                if method == 'denoise':
                    processed = denoise_image(img, data.get('denoise_method', 'bilateral'))
                    results.append({
                        'index': i,
                        'success': True,
                        'result': image_to_base64(processed)
                    })
                else:
                    # 调用相应的检测接口
                    results.append({
                        'index': i,
                        'success': True,
                        'message': '请使用对应的检测接口'
                    })
            except Exception as e:
                results.append({'index': i, 'error': str(e)})
        
        return jsonify({
            'success': True,
            'total': len(images),
            'processed': len([r for r in results if r.get('success')]),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        return jsonify({'error': str(e)}), 500


# ===================== 启动服务 =====================
if __name__ == '__main__':
    logger.info("启动松材线虫病检测系统 API 服务...")
    logger.info(f"模型目录: {MODEL_DIR}")
    
    # 尝试预加载模型
    try:
        models['yolov8'] = load_yolov8_model()
    except:
        pass
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=True)
