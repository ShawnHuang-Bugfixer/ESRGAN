# Real-ESRGAN 项目结构与功能说明（中文）

## 1. 这个项目是做什么的

`Real-ESRGAN` 是一个图像/视频超分与修复项目，目标是把低清、噪声或压缩过的内容恢复成更清晰的高分辨率结果。  
你可以把它理解为：

- 输入：低质量图片或视频
- 输出：放大并增强后的图片或视频

它支持：

- 通用图片超分（如 `RealESRGAN_x4plus`）
- 动漫图片/视频超分（如 `RealESRGAN_x4plus_anime_6B`、`realesr-animevideov3`）
- 可选人脸增强（集成 GFPGAN）
- 训练与微调（基于 `basicsr` 训练框架）

## 2. 目录结构（按用途）

### 核心代码

- `realesrgan/`
- `realesrgan/archs/`：网络结构定义（生成器、判别器等）
- `realesrgan/models/`：模型封装与训练逻辑（RealESRGAN/RealESRNet）
- `realesrgan/data/`：数据集读取与退化数据构造
- `realesrgan/utils.py`：核心推理类 `RealESRGANer`（加载模型、切 tile、推理、后处理）
- `realesrgan/train.py`：训练入口（调用 BasicSR 的 `train_pipeline`）

### 推理入口（你最常用）

- `inference_realesrgan.py`：图片推理入口（支持文件夹批处理、tile、防爆显存、alpha/灰度/16-bit）
- `inference_realesrgan_video.py`：视频推理入口（基于 ffmpeg 读写与封装音视频）

### 配置与脚本

- `options/*.yml`：训练/微调配置（模型结构、数据、损失、优化器等）
- `scripts/`：数据预处理与转换脚本（切图、元信息生成、ONNX 转换等）

### 文档与资源

- `docs/`：训练、模型说明、FAQ、对比文档
- `assets/`：README 图片素材
- `weights/`：预训练权重存放目录

### 测试与样例

- `tests/`：pytest 测试
- `tests/data/`：测试数据（包括普通文件和 lmdb）
- `inputs/`：推理示例输入

## 3. 核心运行流程（简单版）

### 图片推理流程（`inference_realesrgan.py`）

1. 解析参数（输入路径、模型名、放大倍数、tile 等）
2. 根据模型名构建网络并定位/下载权重
3. 创建 `RealESRGANer`
4. 逐张读取输入图片并执行增强
5. 保存到输出目录（默认 `results/`）

### 视频推理流程（`inference_realesrgan_video.py`）

1. 用 ffmpeg 读取视频流（可处理音轨）
2. 帧级别调用 `RealESRGANer.enhance`
3. 把增强帧编码回视频并合成输出

### 训练流程（`realesrgan/train.py` + `options/*.yml`）

1. 准备数据集与元信息
2. 选择配置文件（例如 `options/train_realesrgan_x4plus.yml`）
3. 运行训练入口，交给 BasicSR 管理训练循环

## 4. 你可以先看哪些文件

如果你不熟 Python，建议按这个顺序读：

1. `README.md`：先了解能做什么、如何运行
2. `inference_realesrgan.py`：最短路径理解“如何用”
3. `realesrgan/utils.py`：理解真正的增强逻辑
4. `options/train_realesrgan_x4plus.yml`：理解训练参数怎么组织
5. `realesrgan/train.py`：理解训练入口怎么接到框架

## 5. 常用命令（直接可用）

```bash
# 安装依赖
pip install -r requirements.txt
python setup.py develop

# 图片推理
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results

# 视频推理
python inference_realesrgan_video.py -n realesr-animevideov3 -i inputs/video.mp4 -o results

# 训练（示例）
python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --debug

# 测试
pytest tests/
```

## 6. 一句话总结

这是一个“可直接推理 + 可继续训练”的超分修复工程：  
`inference_*.py` 负责使用，`realesrgan/` 负责核心实现，`options/` 和 `scripts/` 负责训练配置与数据准备。
