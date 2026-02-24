# Python 超分服务 Java 对接文档

## 1. 目的与范围
- 面向 Java 业务端，说明如何与 `python_sr_service` 进行任务投递和结果消费。
- 本文基于当前代码实现，覆盖图片超分链路，不包含视频任务。

## 2. 当前能力边界
- 任务类型：仅支持 `type=image`。
- 结果状态：`RUNNING`、`SUCCEEDED`、`FAILED`。
- 重试机制：Python Worker 内部基于 RabbitMQ 延迟队列自动重试（10s/30s/60s）。
- 幂等策略：仅在终态（成功或最终失败）后标记 `eventId` 已处理。
- 模型切换：当前按 Python 服务配置生效，不按单条任务动态切换。
  - 说明：任务消息里的 `modelName/modelVersion` 会被校验与透传，但推理实际使用 `python_sr_service` 本地配置中的模型。

## 3. RabbitMQ 拓扑约定

### 3.1 任务侧（Java -> Python）
- Exchange：`x.sr.task.direct`（direct）
- Routing Key：`sr.task`
- Queue：`sr.task.queue`
- 持久化：建议消息 `delivery_mode=2`

### 3.2 结果侧（Python -> Java）
- Exchange：`x.sr.result.direct`（direct）
- Routing Key：`sr.result`
- Java 侧建议绑定独立结果队列，例如：`sr.result.java.{env}`

### 3.3 重试侧（Python 内部）
- Exchange：`x.sr.retry.direct`
- Python 内部会声明重试队列：
  - `sr.task.queue.retry.10s`
  - `sr.task.queue.retry.30s`
  - `sr.task.queue.retry.60s`
- Java 业务端不需要直接投递到重试交换机。

## 4. 任务消息协议（Java 发布）

### 4.1 JSON 字段
```json
{
  "schemaVersion": "1.0",
  "eventId": "evt_task_8d9c0f3d9e1f4f26b6c8b8f8b1f1a001",
  "timestamp": "2026-02-23T12:34:56Z",
  "taskId": 100001,
  "taskNo": "SR202602230001",
  "userId": 20001,
  "type": "image",
  "inputFileKey": "input/2026/02/23/demo.png",
  "scale": 4,
  "modelName": "RealESRGAN_x4plus",
  "modelVersion": "v1.0.0",
  "attempt": 1,
  "traceId": "trace_5b0d5b6fd5c649f0a81d0f5b9f7b7abc"
}
```

### 4.2 必填字段
- `schemaVersion`
- `eventId`
- `timestamp`（ISO-8601）
- `taskId`
- `taskNo`
- `userId`
- `type`
- `inputFileKey`
- `scale`
- `modelName`
- `modelVersion`
- `attempt`
- `traceId`

### 4.3 字段约束建议
- `type`：固定为 `image`。
- `eventId`：全局唯一，建议 UUID。
- `traceId`：全链路唯一，贯穿 Java/Python/存储。
- `attempt`：由 Java 业务侧维护业务重试次数，首发为 `1`。
- `inputFileKey`：对象存储中的源图 key，Python 会先下载再推理。
- `scale`：建议使用 `2` 或 `4`。

## 5. 结果消息协议（Java 消费）

### 5.1 JSON 字段
```json
{
  "schemaVersion": "1.0",
  "eventId": "evt_result_3e6b8b7250f54d2eaed3fba8b1a7fd4e",
  "timestamp": "2026-02-23T12:35:08.123456+00:00",
  "taskId": 100001,
  "status": "SUCCEEDED",
  "progress": 100,
  "outputFileKey": "output/100001/SR202602230001_x4.png",
  "costMs": 1187,
  "attempt": 1,
  "errorCode": null,
  "errorMsg": null,
  "traceId": "trace_5b0d5b6fd5c649f0a81d0f5b9f7b7abc"
}
```

### 5.2 状态语义
- `RUNNING`：任务已进入执行。
- `SUCCEEDED`：推理和上传完成，`outputFileKey` 有值。
- `FAILED`：最终失败，`errorCode/errorMsg` 有值。

### 5.3 Java 端处理建议
- 同一个 `taskId` 可能收到多条消息（至少 `RUNNING` + 终态）。
- 以终态（`SUCCEEDED/FAILED`）作为任务完结条件。
- 建议使用 `eventId` 去重消费，避免重复处理。

## 6. 错误码（FAILED 时关注）
- `SCHEMA_INVALID`：消息字段缺失或格式不合法。
- `TYPE_NOT_SUPPORTED`：当前仅支持 `image`。
- `INPUT_NOT_FOUND`：输入图片读取失败。
- `COS_DOWNLOAD_FAILED`：下载输入失败。
- `MODEL_NOT_FOUND`：模型名不支持或权重文件缺失。
- `INFER_RUNTIME_ERROR`：推理运行时错误（可重试场景）。
- `GPU_OOM`：显存不足。
- `COS_UPLOAD_FAILED`：上传输出失败。
- `MYSQL_WRITE_FAILED`：事件写库失败（主流程尽力而为）。
- `INTERNAL_ERROR`：未分类内部错误。

## 7. 模型选择与配置

### 7.1 当前可用模型
- `RealESRGAN_x4plus`
- `RealESRNet_x4plus`
- `RealESRGAN_x4plus_anime_6B`
- `RealESRGAN_x2plus`
- `realesr-animevideov3`
- `realesr-general-x4v3`

### 7.2 服务端配置项（Python）
- `MODEL_NAME`：服务当前使用模型名。
- `MODEL_WEIGHTS`：模型权重路径；为空时默认 `weights/{MODEL_NAME}.pth`。
- `MODEL_DENOISE_STRENGTH`：仅 `realesr-general-x4v3` 有效，范围 `[0,1]`。
  - `1.0`：仅主模型。
  - `<1.0`：自动与 `realesr-general-wdn-x4v3` 组合做 DNI。

### 7.3 CPU/GPU 行为
- `DEVICE=cpu`：强制 CPU 推理。
- `DEVICE=cuda:0`：优先 GPU。
- 若配置 CUDA 但机器无可用 GPU，当前逻辑会自动回退到 CPU。

## 8. Java 端最小接入步骤
1. 创建并绑定任务交换机、任务队列、任务路由键。
2. 业务创建任务时，先上传原图到对象存储，拿到 `inputFileKey`。
3. 按第 4 节组装 JSON，发布到 `x.sr.task.direct` + `sr.task`。
4. Java 消费 `x.sr.result.direct` 的业务结果队列，按第 5 节更新任务状态。
5. 收到 `SUCCEEDED` 后保存 `outputFileKey`，对外提供下载地址。

## 9. Java 端发布/消费示例（伪代码）
```java
// 发布任务
rabbitTemplate.convertAndSend(
    "x.sr.task.direct",
    "sr.task",
    taskJson,
    m -> {
        m.getMessageProperties().setContentType("application/json");
        m.getMessageProperties().setDeliveryMode(MessageDeliveryMode.PERSISTENT);
        return m;
    }
);

// 消费结果
@RabbitListener(queues = "sr.result.java.prod")
public void onResult(String body) {
    ResultMessage msg = parse(body);
    switch (msg.getStatus()) {
        case "RUNNING":
            updateProgress(msg.getTaskId(), msg.getProgress());
            break;
        case "SUCCEEDED":
            markSuccess(msg.getTaskId(), msg.getOutputFileKey(), msg.getCostMs());
            break;
        case "FAILED":
            markFailed(msg.getTaskId(), msg.getErrorCode(), msg.getErrorMsg());
            break;
        default:
            log.warn("unknown status {}", msg.getStatus());
    }
}
```

## 10. 联调检查清单
- RabbitMQ：交换机/队列/路由键与本文一致。
- 对象存储：`inputFileKey` 对应对象可读。
- Python 权重：`weights/` 下存在配置对应模型文件。
- 结果消费：Java 能持续收到 `RUNNING` 与终态消息。
- 幂等：重复投递同一 `eventId` 不应重复执行推理。

## 11. 常见问题
- 收到 `MODEL_NOT_FOUND`：检查 `MODEL_NAME` 与权重路径是否匹配。
- 只收到 `RUNNING` 未收到终态：优先检查对象存储权限、模型加载与推理日志。
- 推理慢或 OOM：减小输入尺寸、启用 `tile`、或切换更轻量模型。

## 12. 版本建议
- 建议 Java 端先按 `schemaVersion=1.0` 对接。
- 后续若协议扩展，采用向后兼容方式新增字段，避免删除已有字段。
