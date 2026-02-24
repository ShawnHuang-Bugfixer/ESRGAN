# Python 超分辨重构服务架构设计（可落地版）

## 1. 范围与目标
- 本文仅覆盖 Python 推理服务（数据面）设计，不包含 Java 控制面实现。
- 目标：可独立部署、可消费任务、可执行图片/视频超分、可回传结果、可观测可重试。

## 2. 服务定位
```text
[RabbitMQ: sr.task.queue] -> [Python SR Worker(GPU)] -> [RabbitMQ: sr.result.queue]
                                   |
                                   +-> [Object Storage: download input / upload output]
                                   +-> [Local Temp Workspace]
                                   +-> [Real-ESRGAN Inference Pipeline]
```

## 3. 运行形态
- 进程模型：单进程 + 单消费通道（每 GPU 1 个 worker，推荐）。
- 并发策略：`prefetch_count=1`，避免单卡显存竞争。
- 部署建议：
  - 单机单卡：1 worker
  - 单机多卡：按 GPU 数起多个 worker，分别绑定 `CUDA_VISIBLE_DEVICES`
  - 多机扩展：同一队列水平扩容

## 4. 模块划分
建议目录（示例）：
```text
python_sr_service/
  app.py                      # 入口
  config.py                   # 配置加载
  worker/
    consumer.py               # MQ 消费
    publisher.py              # MQ 结果发布
  pipeline/
    image_pipeline.py         # 图片流程
    video_pipeline.py         # 视频流程
    phases.py                 # 通用阶段定义
  storage/
    s3_client.py              # 对象存储适配
  idempotency/
    store.py                  # 幂等存储接口
  runtime/
    workspace.py              # 临时目录管理
    progress.py               # 进度上报组装
  domain/
    schema.py                 # 消息模型
    errors.py                 # 错误码
```

## 5. 配置项（最小集）
- RabbitMQ
  - `MQ_URL`
  - `MQ_TASK_QUEUE=sr.task.queue`
  - `MQ_RESULT_EXCHANGE=x.sr.result.direct`
  - `MQ_RESULT_ROUTING_KEY=sr.result`
  - `MQ_PREFETCH=1`
- 对象存储
  - `S3_ENDPOINT`
  - `S3_ACCESS_KEY`
  - `S3_SECRET_KEY`
  - `S3_BUCKET`
- 推理
  - `MODEL_NAME=RealESRGAN_x4plus`
  - `MODEL_WEIGHTS=weights/RealESRGAN_x4plus.pth`
  - `DEVICE=cuda:0`
  - `MAX_VIDEO_FRAMES`（防止超大任务）
- 运行时
  - `WORK_DIR=./runtime
  - `TASK_TIMEOUT_SECONDS`
  - `LOG_LEVEL=INFO`

## 6. 消息契约（Python 侧口径）

### 6.1 任务消息 `sr.task`
```json
{
  "schemaVersion": "1.0",
  "eventId": "evt_task_20260214_0001",
  "timestamp": "2026-02-14T14:00:00Z",
  "taskId": 123456,
  "taskNo": "SR20260214XXXX",
  "userId": 10001,
  "type": "image",
  "inputFileKey": "input/2026/02/a.png",
  "scale": 4,
  "modelName": "Real-ESRGAN",
  "modelVersion": "v1.0.0",
  "attempt": 1,
  "traceId": "9d2f..."
}
```

### 6.2 结果消息 `sr.result`
```json
{
  "schemaVersion": "1.0",
  "eventId": "evt_result_20260214_0001",
  "timestamp": "2026-02-14T14:00:18Z",
  "taskId": 123456,
  "status": "SUCCEEDED",
  "progress": 100,
  "outputFileKey": "output/2026/02/a_x4.png",
  "costMs": 18320,
  "attempt": 1,
  "errorCode": null,
  "errorMsg": null,
  "traceId": "9d2f..."
}
```

状态仅使用：`RUNNING/SUCCEEDED/FAILED/CANCELLED`（控制面维护 `CREATED/QUEUED`）。

## 7. 主流程

### 7.1 消费与执行
1. 消费 `sr.task.queue` 消息并做 schema 校验。
2. 幂等检查（`eventId` 或 `taskId+attempt`）：已处理则直接 ack。
3. 发送一次 `RUNNING` 进度消息（可选，progress=5）。
4. 下载输入文件到本地工作目录。
5. 根据 `type` 分发：
   - `image` -> `image_pipeline`
   - `video` -> `video_pipeline`
6. 上传输出到对象存储，得到 `outputFileKey`。
7. 发布 `SUCCEEDED` 结果并 ack 原消息。

### 7.2 失败处理
- 可重试错误（网络抖动、临时 I/O）：nack/requeue 或投 retry 队列。
- 不可重试错误（参数非法、文件损坏、模型不支持）：发布 `FAILED` 后 ack。
- 异常时保证：
  - 清理临时目录
  - 发布失败结果（包含 `errorCode/errorMsg`）

## 8. 图片/视频管线策略

### 8.1 图片
阶段建议与进度：
- `download` -> 10
- `enhance` -> 70
- `upload` -> 90
- `done` -> 100

### 8.2 视频
对外一个任务，对内三阶段：
- `extract_frames`（10~30）
- `enhance_frames`（30~80）
- `merge_video`（80~95）
- `upload_done`（100）

实现建议：
- `ffmpeg` 抽帧/合成。
- 帧处理支持断点目录扫描（阶段失败可重复执行当前阶段）。
- 限制最大帧数和最长时长，避免任务失控。

## 9. 幂等、重试与错误码

### 9.1 幂等
- 键：优先 `eventId`，退化到 `taskId+attempt`。
- 存储：MVP 可用 sqlite/本地文件；生产建议 redis/mysql。
- 生命周期：保留 24~72 小时去重记录。

### 9.2 重试策略
- 建议由 MQ 层提供多级延迟队列：10s/30s/60s。
- Python 仅区分错误类型并选择：
  - `retryable` -> 触发重试路径
  - `non_retryable` -> 直接失败

### 9.3 错误码建议
- `INPUT_NOT_FOUND`
- `INPUT_INVALID`
- `MODEL_NOT_FOUND`
- `GPU_OOM`
- `FFMPEG_ERROR`
- `UPLOAD_FAILED`
- `INTERNAL_ERROR`

## 10. 可观测与运维
- 日志字段：`traceId/taskId/eventId/attempt/phase/costMs/status`。
- 指标建议：
  - 任务吞吐（qps）
  - 成功率/失败率
  - 各阶段耗时（P50/P95/P99）
  - 队列堆积长度
  - GPU 利用率与显存占用
- 健康检查：
  - `/healthz`：进程存活
  - `/readyz`：模型与 MQ/S3 连接可用

## 11. 安全与资源治理
- 不在消息中传递密钥。
- 凭证通过环境变量或密钥管理系统注入。
- 限制输入文件大小、视频时长、分辨率上限。
- 每任务隔离工作目录，任务结束强制清理。

## 12. MVP 落地清单（Python-only）
1. 消费 `sr.task.queue`。
2. 下载输入文件。
3. 图片超分推理并上传输出。
4. 发布 `sr.result` 成功/失败消息。
5. 基础日志 + 基础错误码。

后续增强：
1. 视频三阶段。
2. 幂等存储持久化。
3. 分级重试与取消检查。
4. 指标与告警接入。
