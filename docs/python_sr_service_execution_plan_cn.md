# Python 超分辨重构服务执行计划（MVP 到可扩展版）

## 1. 摘要
目标是在当前 Real-ESRGAN 仓库中新增一个独立的 Python Worker 服务，消费 `sr.task.queue`，完成图片超分推理，上传结果并发布 `sr.result`。

已锁定决策：
- `Worker-only`
- 对象存储优先接入腾讯云 `COS`（保留 S3 兼容抽象）
- `Redis` 幂等
- MVP 仅图片
- 首版不支持取消任务
- `Java` 管理 `sr_task` 主状态，`Python` 仅负责推理与事件日志

## 2. 交付范围

### 2.1 In Scope（本次实现）
1. 新增 `python_sr_service` 服务骨架与启动入口。
2. RabbitMQ 任务消费与结果发布。
3. 腾讯云 COS 下载输入与上传输出（S3 接口抽象兼容）。
4. MySQL 建库建表与 Python 事件日志落库（`sr_task_event`）。
5. 基于现有 `inference_realesrgan.py` 能力封装图片推理管线。
6. Redis 幂等去重（按 `eventId`）。
7. 结构化日志与基础健康检查。
8. 单元测试 + 最小集成测试（可本地跑）。

### 2.2 Out of Scope（后续阶段）
1. 视频链路（extract/enhance/merge）。
2. 取消任务。
3. 多模型动态路由与优先级队列。
4. 自动扩缩容与复杂调度。

## 3. 代码结构规划
1. 新增目录 `python_sr_service/`。
2. 新增 `python_sr_service/app.py` 作为入口（启动 consumer）。
3. 新增 `python_sr_service/config.py`（环境变量读取与校验）。
4. 新增 `python_sr_service/domain/schema.py`（消息模型与校验）。
5. 新增 `python_sr_service/domain/errors.py`（错误码定义）。
6. 新增 `python_sr_service/worker/consumer.py`（消费、ack/nack、分发）。
7. 新增 `python_sr_service/worker/publisher.py`（结果消息发布）。
8. 新增 `python_sr_service/pipeline/image_pipeline.py`（下载-推理-上传主流程）。
9. 新增 `python_sr_service/storage/cos_client.py`（COS 适配）。
10. 新增 `python_sr_service/storage/object_storage.py`（统一存储接口，兼容 S3/COS）。
11. 新增 `python_sr_service/idempotency/redis_store.py`（去重）。
12. 新增 `python_sr_service/persistence/mysql_event_repo.py`（`sr_task_event` 写入）。
13. 新增 `python_sr_service/runtime/workspace.py`（任务临时目录管理）。
14. 新增 `python_sr_service/runtime/logging.py`（结构化日志初始化）。
15. 新增 `tests/python_sr_service/` 对应测试。

## 4. 接口与契约（实现必须对齐）

### 4.1 输入消息 `sr.task`（必须字段）
1. `schemaVersion` string
2. `eventId` string
3. `timestamp` string(ISO-8601)
4. `taskId` int
5. `taskNo` string
6. `userId` int
7. `type` string（本期仅接受 `image`）
8. `inputFileKey` string
9. `scale` int
10. `modelName` string
11. `modelVersion` string
12. `attempt` int
13. `traceId` string

### 4.2 输出消息 `sr.result`
1. `schemaVersion` string
2. `eventId` string（新生成）
3. `timestamp` string
4. `taskId` int
5. `status` enum（`RUNNING`/`SUCCEEDED`/`FAILED`）
6. `progress` int
7. `outputFileKey` string|null
8. `costMs` int
9. `attempt` int
10. `errorCode` string|null
11. `errorMsg` string|null
12. `traceId` string

### 4.3 错误码最小集合
1. `SCHEMA_INVALID`
2. `TYPE_NOT_SUPPORTED`
3. `INPUT_NOT_FOUND`
4. `COS_DOWNLOAD_FAILED`
5. `MODEL_NOT_FOUND`
6. `INFER_RUNTIME_ERROR`
7. `GPU_OOM`
8. `COS_UPLOAD_FAILED`
9. `MYSQL_WRITE_FAILED`
10. `INTERNAL_ERROR`

## 5. 执行流程设计（单任务）
1. Consumer 收到消息并做 JSON/schema 校验。
2. 若 `type != image`，直接发布 `FAILED(TYPE_NOT_SUPPORTED)` 并 ack。
3. Redis 幂等检查：
   - `SETNX dedupe:{eventId}`，带 TTL（默认 72h）。
   - 已存在则直接 ack，不重复执行。
4. 落库事件 `RECEIVED` 到 `sr_task_event`。
5. 发布 `RUNNING(progress=5)`（可观测）。
6. 创建任务工作目录：`WORK_DIR/{taskId}_{attempt}`。
7. 从 COS 下载 `inputFileKey` 到本地，完成后落库 `DOWNLOADED`。
8. 调用图片推理管线，完成后落库 `INFERRED`。
9. 上传结果到 COS，生成 `outputFileKey`，落库 `UPLOADED`。
10. 发布 `SUCCEEDED(progress=100)`，包含 `costMs`，落库 `SUCCEEDED`。
11. 清理临时目录并 ack。
12. 异常分流：
   - 可判定业务失败：发布 `FAILED` 后 ack，并落库失败事件。
   - 可重试失败：路由至 RabbitMQ 延迟重试队列。

## 6. 配置与运行规范

### 6.1 环境变量
1. `MQ_URL`
2. `MQ_TASK_QUEUE=sr.task.queue`
3. `MQ_RESULT_EXCHANGE=x.sr.result.direct`
4. `MQ_RESULT_ROUTING_KEY=sr.result`
5. `MQ_RETRY_EXCHANGE=x.sr.retry.direct`
6. `MQ_PREFETCH=1`
7. `COS_SECRET_ID`
8. `COS_SECRET_KEY`
9. `COS_REGION`
10. `COS_BUCKET`
11. `COS_SCHEME=https`
12. `REDIS_URL`
13. `MYSQL_DSN`
14. `IDEMP_TTL_SECONDS=259200`
15. `MODEL_NAME=RealESRGAN_x4plus`
16. `MODEL_WEIGHTS=weights/RealESRGAN_x4plus.pth`
17. `DEVICE=cuda:0`
18. `WORK_DIR=./runtime`
19. `LOG_LEVEL=INFO`

### 6.2 运行命令
1. 本地开发：`python -m python_sr_service.app`
2. 生产建议：按 GPU 启多个进程，每进程固定一个 `DEVICE`。

## 7. 测试计划与验收标准

### 7.1 单元测试
1. schema 校验成功/失败路径。
2. Redis 幂等：首次处理、重复消息跳过、TTL 行为。
3. COS 客户端异常映射为统一错误码。
4. MySQL 事件写入成功/失败路径。
5. 结果消息序列化与字段完整性。

### 7.2 集成测试（可 mock MQ/COS/Redis/MySQL）
1. 正常图片任务：收到 `RUNNING` + `SUCCEEDED`。
2. 输入文件不存在：收到 `FAILED(INPUT_NOT_FOUND)`。
3. 重复 `eventId`：仅第一次触发推理。
4. 非 image 任务：快速失败且不推理。
5. COS 暂时失败：进入 retry 队列后回投。

### 7.3 验收标准（DoD）
1. 能稳定消费 `sr.task.queue` 且无阻塞。
2. 同一 `eventId` 不会重复推理。
3. 成功任务产出可下载结果文件并发布 `SUCCEEDED`。
4. `sr_task_event` 可按 `taskId/traceId` 还原执行链路。
5. 失败任务必定发布 `FAILED` 且错误码可解析。

## 8. 分阶段执行计划（四阶段整合）

### Phase 1：接入腾讯云 COS（文件管理）
目标：完成输入下载/输出上传能力。

交付项：
1. `cos_client.py` 与统一存储接口 `object_storage.py`。
2. COS 连接配置与鉴权（`COS_SECRET_ID/SECRET_KEY/REGION/BUCKET`）。
3. 下载/上传 API：`download(object_key, local_path)`、`upload(local_path, object_key)`。
4. 错误码映射：`COS_DOWNLOAD_FAILED`、`COS_UPLOAD_FAILED`。

验收：
1. 能上传并回读样例图片。
2. 非法 key、网络异常能返回标准错误码。

### Phase 2：接入 MySQL（建库建表）
目标：完成数据库初始化和事件审计落库。


交付项：
1. 建库与 DDL：`sr_task`、`sr_task_event`。
2. Python 持久化层：仅写 `sr_task_event`，不写 `sr_task.status`。
3. 事件类型规范：`RECEIVED/DOWNLOADED/INFERRED/UPLOADED/SUCCEEDED/FAILED`。

验收：
1. 单任务完成后可在 `sr_task_event` 查到完整阶段记录。
2. 写库失败可观测并返回 `MYSQL_WRITE_FAILED`。

### Phase 3：接入 RabbitMQ（消息收发与重试）
目标：完成消息驱动闭环和重试机制。

账号密码：admin/Admin@123 

rabbitmq 服务地址：localhost

redis 信息：
服务地址：localhost
密码：qwDFerAs1

交付项：
1. 消费 `sr.task.queue`，发布 `sr.result`。
2. 消费确认机制：手动 ack，`prefetch_count=1`。
3. 延迟重试队列：10s/30s/60s + DLQ。
4. Redis 幂等：按 `eventId` 去重。



验收：
1. 可发送任务消息并收到结果消息。
2. 可重试异常进入 retry 后自动回投。
3. 重复消息不会重复推理。

### Phase 4：打通超分服务主链路（图片）
目标：形成端到端生产最小闭环。

交付项：
1. 集成 Real-ESRGAN 图片推理 pipeline。
2. 主流程：消费 -> COS 下载 -> 推理 -> COS 上传 -> 结果回传 -> 事件落库。
3. 结构化日志和基础健康检查。

验收：
1. `sr.task` 到 `sr.result(SUCCEEDED)` 全链路打通。
2. 结果文件可在 COS 下载查看。
3. 失败场景（输入缺失/OOM/上传失败）行为符合预期。

## 9. 风险与对策
1. GPU OOM 风险。
   - 对策：默认支持 `tile` 参数并在 OOM 时返回 `GPU_OOM`。
2. 权重文件缺失。
   - 对策：启动时检查 `MODEL_WEIGHTS` 是否存在，失败即不就绪。
3. COS 网络抖动。
   - 对策：下载/上传加有限次重试与超时设置。
4. MQ 连接中断。
   - 对策：consumer 自动重连，发布失败记录错误并触发告警。
5. MySQL 写入抖动。
   - 对策：事件写入失败不阻塞主链路，但必须记录告警日志并上报监控。

## 10. 明确假设与默认值
1. 上游会提供合法的 `sr.task` JSON 并包含 `eventId`。
2. MQ 拓扑可由平台预建，或由 worker 启动时自动声明。
3. Redis、MySQL、COS 在 worker 网络可达。
4. 本期只接受 `type=image`，视频请求直接失败。
5. 任务重试由 RabbitMQ retry 拓扑承担，不由业务代码循环重试。
