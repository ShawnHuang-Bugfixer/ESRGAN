# 超分辨率处理模块架构设计（可落地版，修正版）

## 1. 目标与边界
- 将 Real-ESRGAN 项目抽象为独立的超分辨率重构服务。
- 约束：业务服务（Java）不做推理计算，推理由独立 Python 服务承担。
- 目标：在保证吞吐的同时，确保任务一致性、可追踪、可重试、可取消。

## 2. 总体架构
```text
[Client]
   |
   v
[Java API / 业务服务（控制面）]
   | 1. 上传原文件到对象存储
   | 2. 创建任务记录（DB）
   | 3. 写 Outbox 事件（同事务）
   v
[Outbox Relay]
   | 4. 投递任务消息（RabbitMQ: sr.task）
   v
[RabbitMQ]
   |
   v
[Python 推理服务（数据面, GPU）]
   | 1. 消费任务
   | 2. 下载输入文件（对象存储）
   | 3. 超分推理（图像/视频）
   | 4. 上传输出文件（对象存储）
   | 5. 发布结果消息（RabbitMQ: sr.result）
   v
[Java 结果消费者]
   | 1. 幂等消费结果消息
   | 2. 严格状态流转校验并更新任务状态（DB）
   | 3. 写通知消息并 WebSocket/SSE 推送
   v
[Client]
```

## 3. 组件职责
### 3.1 Java 业务服务（控制面）
- 鉴权、配额与参数校验。
- 任务创建、状态查询、取消任务。
- 维护任务状态机与重试计数。
- 消费推理结果并触发消息推送。
- 维护 Outbox（与任务写入同事务）。

### 3.2 Outbox Relay（建议独立进程或定时任务）
- 扫描未投递 outbox 记录。
- 投递 `sr.task` 消息并记录投递结果。
- 支持失败重试、告警与补偿。

### 3.3 Python 推理服务（数据面）
- 从 `sr.task` 消费任务。
- 执行模型推理：
  - 图片：Real-ESRGAN。
  - 视频：`抽帧 -> 帧超分 -> 合成`。
- 回传结果消息到 `sr.result`。
- 在阶段边界检查取消请求并中止。

### 3.4 对象存储（S3/MinIO）
- 输入文件、输出文件统一管理。
- 消息只传 `objectKey`，不传二进制。

### 3.5 RabbitMQ
- 解耦业务端和推理端。
- 提供重试、死信和削峰能力。

## 4. 任务状态机（统一口径）
状态：
- `CREATED`：任务已创建（尚未成功入队）
- `QUEUED`：任务已入队（等待执行）
- `RUNNING`：推理执行中
- `SUCCEEDED`：成功
- `FAILED`：失败且超过重试上限
- `CANCELLED`：已取消

状态流转：
```text
CREATED -> QUEUED -> RUNNING -> SUCCEEDED
                      \-> FAILED
CREATED/QUEUED/RUNNING -> CANCELLED
```

约束：
- 只允许“前向流转”，禁止旧事件覆盖新状态。
- `SUCCEEDED/FAILED/CANCELLED` 为终态，不可再转移。

## 5. 数据库模型建议
### 5.1 `sr_task`
- `id` bigint PK
- `task_no` varchar(64) unique
- `user_id` bigint not null
- `input_file_key` varchar(512) not null
- `output_file_key` varchar(512) null
- `task_type` varchar(16) not null (`image`/`video`)
- `scale` int not null
- `model_name` varchar(64) not null
- `model_version` varchar(32) not null
- `status` varchar(16) not null
- `progress` int default 0
- `retry_count` int default 0
- `max_retry` int default 3
- `cancel_requested` tinyint/bool default 0
- `cancel_requested_at` datetime null
- `cancelled_at` datetime null
- `error_code` varchar(64) null
- `error_msg` varchar(1024) null
- `trace_id` varchar(64) null
- `created_at` datetime
- `updated_at` datetime

索引建议：
- `uk_task_no(task_no)`
- `idx_user_created(user_id, created_at desc)`
- `idx_status_created(status, created_at asc)`
- `idx_status_updated(status, updated_at asc)`（调度/超时扫描）

### 5.2 `sr_task_event`（建议保留）
- `id` bigint PK
- `task_id` bigint not null
- `event_id` varchar(64) not null
- `event_type` varchar(32) not null
- `event_payload` json/text
- `created_at` datetime

用途：审计、排障、回放。

### 5.3 `sr_outbox`（强烈建议）
- `id` bigint PK
- `aggregate_type` varchar(32) not null（如 `sr_task`）
- `aggregate_id` bigint not null
- `event_type` varchar(32) not null（如 `TASK_CREATED`）
- `event_id` varchar(64) unique
- `payload` json/text not null
- `status` varchar(16) not null（`NEW`/`SENT`/`FAILED`）
- `retry_count` int default 0
- `next_retry_at` datetime null
- `created_at` datetime
- `updated_at` datetime

## 6. RabbitMQ 拓扑建议
交换机：
- `x.sr.task.direct`
- `x.sr.result.direct`
- `x.sr.retry.direct`
- `x.sr.dlx.direct`

队列：
- `sr.task.queue`
- `sr.result.queue`
- `sr.retry.10s.queue`
- `sr.retry.30s.queue`
- `sr.retry.60s.queue`
- `sr.dlq.queue`

路由：
- Java Outbox Relay -> `x.sr.task.direct` -> `sr.task.queue`
- Python -> `x.sr.result.direct` -> `sr.result.queue`
- 失败重试 -> `x.sr.retry.direct` -> `sr.retry.*.queue`（TTL 到期后回投 `x.sr.task.direct`）
- 超过阈值 -> `x.sr.dlx.direct` -> `sr.dlq.queue`

建议：
- Python Worker 设置 `prefetch_count=1`（GPU 任务常见配置）。
- 显式处理 `x-death`，防止无限回投。

## 7. 消息协议（建议）
通用字段建议：
- `schemaVersion`：协议版本
- `eventId`：事件唯一 ID（幂等键）
- `timestamp`：事件时间（ISO-8601）
- `traceId`：链路追踪

### 7.1 任务消息 `sr.task`
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

### 7.2 结果消息 `sr.result`
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

## 8. 一致性、幂等与容错策略
### 8.1 创建任务一致性
- Java 在同一事务内写入 `sr_task(CREATED)` + `sr_outbox(TASK_CREATED)`。
- Outbox Relay 异步投递 MQ，投递成功后标记 `SENT`。
- 避免“任务入库成功但消息丢失”。

### 8.2 消费幂等
- Python 消费 `sr.task`：按 `eventId` 去重（本地缓存 + 持久化幂等记录）。
- Java 消费 `sr.result`：按 `eventId` 去重，并校验状态流转合法性。
- 拒绝旧 `attempt` 或旧状态结果覆盖新状态。

### 8.3 失败重试
- 建议指数/分级退避：`10s -> 30s -> 60s`。
- 超过 `max_retry`：标记 `FAILED` 并投递 `sr.dlq.queue`。

### 8.4 取消任务
- 取消请求到达后，Java 设置 `cancel_requested=1`。
- Python 在阶段边界（下载前、推理前、上传前）检查并尽快退出。
- 若退出成功且任务未终态，更新为 `CANCELLED`。
- 若取消时任务已终态，保持原终态（`SUCCEEDED/FAILED`）。

## 9. 图片/视频执行策略
### 9.1 图片
- 单任务直接推理并上传结果。
- 进度：`10 -> 60 -> 90 -> 100`（下载/推理/上传/完成）。

### 9.2 视频
- 对外保持单 `taskId`，对内阶段化执行：
  - `extract_frames`
  - `enhance_frames`
  - `merge_video`
- 建议记录 `phase`、每阶段耗时、阶段错误码。

## 10. 安全与治理
- 消息中不传签名密钥和临时凭证。
- 对象存储访问使用服务端凭证或短期 STS。
- 限制单用户并发任务数、单文件大小、单任务最大时长。
- 记录 `traceId/taskId/userId/modelVersion` 全链路日志。
- 临时文件（尤其视频帧）设置生命周期清理策略。

## 11. 观测与运维指标（建议补充）
- 任务成功率、失败率、取消率。
- 队列堆积长度、消息滞留时长、DLQ 增长速率。
- 任务耗时分布（P50/P95/P99）。
- GPU 利用率、显存占用、单卡并发数。
- 各阶段耗时（下载/推理/上传/视频阶段）。

## 12. 实施里程碑
### Phase 1（MVP，图片）
- Java：创建任务、状态查询、Outbox 投递。
- RabbitMQ：`sr.task/sr.result` 单队列。
- Python：图片超分全链路跑通。

### Phase 2（稳定性）
- 多级重试队列 + DLQ + 幂等去重。
- 任务取消、失败可观测性、报警。
- 状态机守卫与结果覆盖保护。

### Phase 3（视频与扩展）
- 视频全链路。
- 多模型路由、优先级队列、配额与计费。
- 更细粒度的资源治理与调度策略。
