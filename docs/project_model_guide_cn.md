# Real-ESRGAN 项目模型清单（含说明）

本文汇总当前仓库中的模型，并说明适用场景、推理入口支持情况及本地权重现状。

## 1. 在线模型（Model Zoo）

来源：`docs/model_zoo.md`

### 1.1 通用图像模型

1. `RealESRGAN_x4plus`
- 放大倍率：`x4`
- 说明：通用图像超分主力模型，效果均衡。

2. `RealESRGAN_x2plus`
- 放大倍率：`x2`
- 说明：通用图像 x2 超分模型。

3. `RealESRNet_x4plus`
- 放大倍率：`x4`
- 说明：偏 MSE 路线，结果更平滑。

4. `ESRGAN_SRx4_DF2KOST_official-ff704c30`（官方 ESRGAN x4）
- 放大倍率：`x4`
- 说明：官方 ESRGAN 模型，兼容对比用途较多。

5. `realesr-general-x4v3`
- 放大倍率：`x4`（也可用于 `x1/x2/x3`）
- 说明：轻量模型，显存/速度开销更低，去模糊和去噪能力相对温和。

### 1.2 动漫图像模型

1. `RealESRGAN_x4plus_anime_6B`
- 放大倍率：`x4`
- 说明：针对动漫/插画优化，网络更小（6 个 RRDB block）。

### 1.3 动漫视频模型

1. `realesr-animevideov3`
- 放大倍率：`x4`（也可用于 `x1/x2/x3`）
- 说明：面向动漫视频场景的轻量模型。

### 1.4 训练/微调判别器模型（非直接超分输出模型）

1. `RealESRGAN_x4plus_netD`
2. `RealESRGAN_x2plus_netD`
3. `RealESRGAN_x4plus_anime_6B_netD`

说明：以上 `*_netD` 为判别器，主要用于训练或微调，不是常规推理直接输出模型。

## 2. 项目推理脚本支持的模型

### 2.1 图片推理脚本

入口：`inference_realesrgan.py`

支持：
1. `RealESRGAN_x4plus`
2. `RealESRNet_x4plus`
3. `RealESRGAN_x4plus_anime_6B`
4. `RealESRGAN_x2plus`
5. `realesr-animevideov3`
6. `realesr-general-x4v3`

补充：`realesr-general-x4v3` 支持 `denoise_strength`，会和 `realesr-general-wdn-x4v3` 进行 DNI 混合。

### 2.2 视频推理脚本

入口：`inference_realesrgan_video.py`

支持：
1. `realesr-animevideov3`
2. `RealESRGAN_x4plus_anime_6B`
3. `RealESRGAN_x4plus`
4. `RealESRNet_x4plus`
5. `RealESRGAN_x2plus`
6. `realesr-general-x4v3`

## 3. Python SR 服务（python_sr_service）当前支持范围

入口：`python_sr_service/pipeline/image_pipeline.py`

当前仅支持：
1. `RealESRGAN_x4plus`
2. `RealESRGAN_x2plus`
3. `realesr-animevideov3`

说明：
- 若配置了其他模型名，会抛出 `MODEL_NOT_FOUND`。
- 当使用 `realesr-general-x4v3` 且 `denoise_strength != 1` 时，会自动组合 `realesr-general-wdn-x4v3` 做 DNI。
- 可通过扩展 `python_sr_service/pipeline/image_pipeline.py` 的 `_build_model` 分支增加模型支持。

## 4. 本地权重现状（weights 目录）

当前仓库 `weights/` 下已存在：
1. `RealESRGAN_x4plus.pth`
2. `realesr-animevideov3.pth`

说明：其他模型可由推理脚本按需自动下载，或手动放入 `weights/` 目录。

## 5. 选型建议（简要）

1. 通用照片优先：`RealESRGAN_x4plus`
2. 只需 x2 放大：`RealESRGAN_x2plus`
3. 动漫图像优先：`RealESRGAN_x4plus_anime_6B`
4. 动漫视频优先：`realesr-animevideov3`
5. 资源受限（显存/时延敏感）：`realesr-general-x4v3`

## 6. 配置示例（python_sr_service）

文件：`python_sr_service/application.yml`

```yaml
inference:
  model_name: RealESRGAN_x4plus
  model_weights: ""
  device: cuda:0
  tile: 0
  tile_pad: 10
  pre_pad: 0
  fp32: false
```

环境变量可覆盖：
- `MODEL_NAME`
- `MODEL_WEIGHTS`
- `DEVICE`
- `MODEL_TILE`
- `MODEL_TILE_PAD`
- `MODEL_PRE_PAD`
- `MODEL_FP32`


