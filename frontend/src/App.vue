<template>
  <div class="app-container">
    <!-- 头部 -->
    <header class="header">
      <div class="header-content">
        <h1 class="title">
          <span class="icon">🦠</span>
          基于CT的松材线虫病检测系统
        </h1>
        <p class="subtitle">CT Image Pine Wood Nematode Detection System</p>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="main-content">
      <el-row :gutter="20">
        <!-- 左侧: 上传和模型选择 -->
        <el-col :span="8">
          <el-card class="upload-card">
            <template #header>
              <div class="card-header">
                <span>📤 图像上传</span>
              </div>
            </template>
            
            <!-- 上传区域 -->
            <el-upload
              class="upload-area"
              drag
              :auto-upload="false"
              :show-file-list="false"
              accept="image/*"
              @change="handleFileChange"
            >
              <div v-if="!previewImage">
                <el-icon class="upload-icon"><upload-filled /></el-icon>
                <div class="upload-text">
                  <span>拖拽图像到此处或</span>
                  <em>点击上传</em>
                </div>
                <div class="upload-hint">支持 PNG, JPG, BMP 格式</div>
              </div>
              <div v-else class="preview-container">
                <img :src="previewImage" alt="预览" class="preview-image" />
              </div>
            </el-upload>

            <!-- 手动输入 Base64 -->
            <el-input
              v-model="base64Input"
              type="textarea"
              :rows="3"
              placeholder="或粘贴 Base64 编码的图像..."
              class="base64-input"
            />
            <el-button type="primary" @click="loadBase64" :disabled="!base64Input">
              加载图像
            </el-button>
          </el-card>

          <!-- 模型选择 -->
          <el-card class="model-card">
            <template #header>
              <div class="card-header">
                <span>🤖 模型选择</span>
              </div>
            </template>
            
            <el-radio-group v-model="selectedModel" class="model-group">
              <el-radio-button label="yolov8">YOLOv8</el-radio-button>
              <el-radio-button label="unet">U-Net</el-radio-button>
              <el-radio-button label="maskrcnn">Mask R-CNN</el-radio-button>
            </el-radio-group>

            <div class="model-info">
              <el-tag v-if="selectedModel === 'yolov8'" type="success">
                YOLOv8: 高效目标检测模型，适合实时应用
              </el-tag>
              <el-tag v-else-if="selectedModel === 'unet'" type="warning">
                U-Net: 医学图像分割经典模型
              </el-tag>
              <el-tag v-else type="danger">
                Mask R-CNN: 实例分割模型，可区分重叠目标
              </el-tag>
            </div>
          </el-card>

          <!-- 参数设置 -->
          <el-card class="params-card">
            <template #header>
              <div class="card-header">
                <span>⚙️ 参数设置</span>
              </div>
            </template>
            
            <el-form label-width="100px">
              <el-form-item label="置信度阈值">
                <el-slider 
                  v-model="confidence" 
                  :min="0" 
                  :max="1" 
                  :step="0.05"
                  :format-tooltip="val => val.toFixed(2)"
                />
                <span class="confidence-value">{{ confidence.toFixed(2) }}</span>
              </el-form-item>
              
              <el-form-item label="去噪方法">
                <el-select v-model="denoiseMethod" placeholder="选择去噪方法">
                  <el-option label="双边滤波" value="bilateral" />
                  <el-option label="高斯滤波" value="gaussian" />
                  <el-option label="中值滤波" value="median" />
                </el-select>
              </el-form-item>
            </el-form>
          </el-card>
        </el-col>

        <!-- 中间: 结果展示 -->
        <el-col :span="10">
          <el-card class="result-card">
            <template #header>
              <div class="card-header">
                <span>📊 检测结果</span>
                <el-tag v-if="resultCount > 0" type="success" size="large">
                  检测到 {{ resultCount }} 个目标
                </el-tag>
              </div>
            </template>
            
            <div class="result-container">
              <img 
                v-if="resultImage" 
                :src="resultImage" 
                alt="检测结果" 
                class="result-image"
              />
              <div v-else class="result-placeholder">
                <el-icon class="placeholder-icon"><picture /></el-icon>
                <p>上传图像后开始检测</p>
              </div>
            </div>

            <!-- 检测详情 -->
            <div v-if="detections.length > 0" class="detections-list">
              <h4>检测详情:</h4>
              <el-table :data="detections" stripe>
                <el-table-column prop="index" label="序号" width="60" />
                <el-table-column prop="label" label="类别" width="120" />
                <el-table-column prop="confidence" label="置信度">
                  <template #default="scope">
                    {{ (scope.row.confidence * 100).toFixed(1) }}%
                  </template>
                </el-table-column>
                <el-table-column prop="bbox" label="边界框">
                  <template #default="scope">
                    {{ formatBbox(scope.row.bbox) }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </el-card>
        </el-col>

        <!-- 右侧: 功能按钮和历史 -->
        <el-col :span="6">
          <!-- 操作按钮 -->
          <el-card class="action-card">
            <template #header>
              <div class="card-header">
                <span>🚀 操作</span>
              </div>
            </template>
            
            <div class="action-buttons">
              <el-button 
                type="primary" 
                size="large" 
                :icon="VideoPlay" 
                :loading="processing"
                :disabled="!previewImage"
                @click="runDetection"
                class="action-btn"
              >
                开始检测
              </el-button>
              
              <el-button 
                size="large" 
                :icon="Brush" 
                :loading="denoising"
                :disabled="!previewImage"
                @click="runDenoise"
                class="action-btn"
              >
                去噪处理
              </el-button>
              
              <el-button 
                size="large" 
                :icon="Download" 
                :disabled="!resultImage"
                @click="downloadResult"
                class="action-btn"
              >
                下载结果
              </el-button>
              
              <el-button 
                size="large" 
                :icon="RefreshLeft" 
                @click="resetAll"
                class="action-btn"
              >
                重置
              </el-button>
            </div>
          </el-card>

          <!-- 模型状态 -->
          <el-card class="status-card">
            <template #header>
              <div class="card-header">
                <span>📈 模型状态</span>
              </div>
            </template>
            
            <el-descriptions :column="1" border>
              <el-descriptions-item label="API 状态">
                <el-tag :type="apiStatus === 'connected' ? 'success' : 'danger'">
                  {{ apiStatus === 'connected' ? '已连接' : '未连接' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="YOLOv8">
                <el-tag :type="modelStatus.yolov8 ? 'success' : 'info'">
                  {{ modelStatus.yolov8 ? '已加载' : '未加载' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="U-Net">
                <el-tag :type="modelStatus.unet ? 'success' : 'info'">
                  {{ modelStatus.unet ? '已加载' : '未加载' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="Mask R-CNN">
                <el-tag :type="modelStatus.maskrcnn ? 'success' : 'info'">
                  {{ modelStatus.maskrcnn ? '已加载' : '未加载' }}
                </el-tag>
              </el-descriptions-item>
            </el-descriptions>
            
            <el-button 
              type="info" 
              @click="checkApiStatus"
              class="check-btn"
            >
              刷新状态
            </el-button>
          </el-card>

          <!-- 帮助信息 -->
          <el-card class="help-card">
            <template #header>
              <div class="card-header">
                <span>❓ 使用说明</span>
              </div>
            </template>
            
            <el-steps direction="vertical" :space="60" :active="4">
              <el-step title="上传图像" description="拖拽或选择CT图像文件" />
              <el-step title="选择模型" description="选择检测模型(YOLOv8/U-Net/Mask R-CNN)" />
              <el-step title="调整参数" description="设置置信度和去噪方法" />
              <el-step title="开始检测" description="点击检测按钮获取结果" />
            </el-steps>
          </el-card>
        </el-col>
      </el-row>
    </main>

    <!-- 底部 -->
    <footer class="footer">
      <p>© 2025 东北林业大学 - 基于CT的松材线虫病检测系统 | 计算机与控制工程学院</p>
      <p>指导教师: 邱兆文 教授 | 学生: 刘昕月</p>
    </footer>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled, Picture, Download, RefreshLeft, VideoPlay, Brush } from '@element-plus/icons-vue'
import axios from 'axios'

const API_BASE = 'http://localhost:5000/api'

// 状态变量
const previewImage = ref('')
const resultImage = ref('')
const base64Input = ref('')
const selectedModel = ref('yolov8')
const confidence = ref(0.5)
const denoiseMethod = ref('bilateral')
const processing = ref(false)
const denoising = ref(false)
const detections = ref([])
const resultCount = ref(0)
const apiStatus = ref('disconnected')
const modelStatus = reactive({
  yolov8: false,
  unet: false,
  maskrcnn: false
})

// 处理文件上传
const handleFileChange = (uploadFile) => {
  const file = uploadFile.raw
  if (!file) return
  
  const reader = new FileReader()
  reader.onload = (e) => {
    previewImage.value = e.target.result
    resultImage.value = ''
    detections.value = []
    resultCount.value = 0
  }
  reader.readAsDataURL(file)
}

// 从 Base64 加载图像
const loadBase64 = () => {
  if (!base64Input.value) return
  
  try {
    // 移除可能的 data URL 前缀
    let base64 = base64Input.value
    if (base64.includes(',')) {
      base64 = base64.split(',')[1]
    }
    
    previewImage.value = `data:image/png;base64,${base64}`
    resultImage.value = ''
    detections.value = []
    resultCount.value = 0
    
    ElMessage.success('图像加载成功')
  } catch (e) {
    ElMessage.error('图像加载失败: ' + e.message)
  }
}

// 图像转 Base64
const imageToBase64 = (dataUrl) => {
  return dataUrl.split(',')[1] || dataUrl
}

// 运行检测
const runDetection = async () => {
  if (!previewImage.value) {
    ElMessage.warning('请先上传图像')
    return
  }
  
  processing.value = true
  detections.value = []
  resultCount.value = 0
  
  try {
    const imageBase64 = imageToBase64(previewImage.value)
    
    let endpoint = '/detect/yolov8'
    if (selectedModel.value === 'unet') {
      endpoint = '/detect/unet'
    } else if (selectedModel.value === 'maskrcnn') {
      endpoint = '/detect/maskrcnn'
    }
    
    const response = await axios.post(`${API_BASE}${endpoint}`, {
      image: imageBase64,
      confidence: confidence.value
    })
    
    if (response.data.success) {
      resultImage.value = `data:image/png;base64,${response.data.result}`
      resultCount.value = response.data.count || 0
      
      if (response.data.detections) {
        detections.value = response.data.detections.map((d, i) => ({
          ...d,
          index: i + 1
        }))
      }
      
      ElMessage.success(`检测完成，发现 ${resultCount.value} 个目标`)
    } else {
      ElMessage.error(response.data.error || '检测失败')
    }
  } catch (e) {
    console.error('检测失败:', e)
    ElMessage.error('检测失败: ' + (e.response?.data?.error || e.message))
  } finally {
    processing.value = false
  }
}

// 去噪处理
const runDenoise = async () => {
  if (!previewImage.value) {
    ElMessage.warning('请先上传图像')
    return
  }
  
  denoising.value = true
  
  try {
    const imageBase64 = imageToBase64(previewImage.value)
    
    const response = await axios.post(`${API_BASE}/denoise`, {
      image: imageBase64,
      method: denoiseMethod.value
    })
    
    if (response.data.success) {
      resultImage.value = `data:image/png;base64,${response.data.result}`
      ElMessage.success('去噪处理完成')
    } else {
      ElMessage.error(response.data.error || '去噪失败')
    }
  } catch (e) {
    console.error('去噪失败:', e)
    ElMessage.error('去噪失败: ' + (e.response?.data?.error || e.message))
  } finally {
    denoising.value = false
  }
}

// 下载结果
const downloadResult = () => {
  if (!resultImage.value) {
    ElMessage.warning('没有可下载的结果')
    return
  }
  
  const link = document.createElement('a')
  link.href = resultImage.value
  link.download = `detection_result_${Date.now()}.png`
  link.click()
  
  ElMessage.success('开始下载')
}

// 重置
const resetAll = () => {
  previewImage.value = ''
  resultImage.value = ''
  base64Input.value = ''
  detections.value = []
  resultCount.value = 0
  ElMessage.info('已重置')
}

// 检查 API 状态
const checkApiStatus = async () => {
  try {
    const response = await axios.get(`${API_BASE}/health`)
    if (response.data.status === 'healthy') {
      apiStatus.value = 'connected'
      ElMessage.success('API 服务已连接')
      
      // 获取模型状态
      const modelResponse = await axios.get(`${API_BASE}/models`)
      if (modelResponse.data.models) {
        modelResponse.data.models.forEach(m => {
          if (m.name === 'YOLOv8') modelStatus.yolov8 = m.available
          else if (m.name === 'U-Net') modelStatus.unet = m.available
          else if (m.name === 'Mask R-CNN') modelStatus.maskrcnn = m.available
        })
      }
    }
  } catch (e) {
    apiStatus.value = 'disconnected'
    ElMessage.warning('API 服务未连接，请确保后端已启动')
  }
}

// 格式化边界框
const formatBbox = (bbox) => {
  if (!bbox) return '-'
  return `[${bbox[0].toFixed(0)}, ${bbox[1].toFixed(0)}, ${bbox[2].toFixed(0)}, ${bbox[3].toFixed(0)}]`
}

// 页面加载时检查 API 状态
onMounted(() => {
  checkApiStatus()
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: white;
  padding: 20px 0;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.header-content {
  text-align: center;
}

.title {
  font-size: 28px;
  margin-bottom: 8px;
}

.icon {
  margin-right: 10px;
}

.subtitle {
  font-size: 14px;
  color: #a0a0a0;
}

.main-content {
  flex: 1;
  padding: 20px;
  max-width: 1600px;
  margin: 0 auto;
  width: 100%;
}

.card-header {
  font-weight: bold;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.upload-card,
.model-card,
.params-card,
.result-card,
.action-card,
.status-card,
.help-card {
  margin-bottom: 20px;
  border-radius: 10px;
}

.upload-area {
  width: 100%;
  margin-bottom: 15px;
}

.upload-icon {
  font-size: 50px;
  color: #409eff;
  margin-bottom: 10px;
}

.upload-text {
  color: #606266;
  margin-bottom: 5px;
}

.upload-text em {
  color: #409eff;
  font-style: normal;
}

.upload-hint {
  font-size: 12px;
  color: #909399;
}

.preview-container {
  width: 100%;
  max-height: 300px;
  overflow: hidden;
  border-radius: 5px;
}

.preview-image {
  width: 100%;
  height: auto;
  object-fit: contain;
}

.base64-input {
  margin-bottom: 10px;
}

.model-group {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.model-group :deep(.el-radio-button) {
  margin-bottom: 10px;
}

.model-info {
  margin-top: 15px;
}

.confidence-value {
  margin-left: 10px;
  font-weight: bold;
  color: #409eff;
}

.result-container {
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  border-radius: 10px;
  overflow: hidden;
}

.result-image {
  max-width: 100%;
  max-height: 400px;
  object-fit: contain;
}

.result-placeholder {
  text-align: center;
  color: #909399;
}

.placeholder-icon {
  font-size: 80px;
  margin-bottom: 15px;
}

.detections-list {
  margin-top: 20px;
}

.detections-list h4 {
  margin-bottom: 10px;
  color: #303133;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.action-btn {
  width: 100%;
}

.check-btn {
  width: 100%;
  margin-top: 15px;
}

.footer {
  background: #1a1a2e;
  color: #a0a0a0;
  text-align: center;
  padding: 20px;
  margin-top: auto;
}

.footer p {
  margin: 5px 0;
}
</style>
