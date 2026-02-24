import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import pika
from pika import exceptions as pika_exceptions

from python_sr_service.config import Settings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.domain.schema import TaskMessage
from python_sr_service.idempotency.redis_store import RedisIdempotencyStore
from python_sr_service.persistence.mysql_event_repo import MySQLEventRepository, TaskEventRecord
from python_sr_service.pipeline.image_pipeline import ImagePipeline
from python_sr_service.runtime.logging import format_log_fields
from python_sr_service.runtime.workspace import WorkspaceManager
from python_sr_service.storage.cos_client import TencentCOSClient
from python_sr_service.worker.publisher import RabbitMQResultPublisher

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    # 重试分层通过延迟队列实现（TTL + 死信回流主队列）。
    RETRY_DELAYS_SECONDS = (10, 30, 60)

    def __init__(
        self,
        settings: Settings,
        idempotency_store: RedisIdempotencyStore,
        publisher: RabbitMQResultPublisher,
        storage: Optional[TencentCOSClient] = None,
        event_repo: Optional[MySQLEventRepository] = None,
        image_pipeline: Optional[ImagePipeline] = None,
        workspace_manager: Optional[WorkspaceManager] = None,
    ):
        self._settings = settings
        self._mq = settings.rabbitmq
        self._idempotency_store = idempotency_store
        self._publisher = publisher
        self._storage = storage or TencentCOSClient(settings.cos)
        self._event_repo = event_repo or MySQLEventRepository(settings.mysql)
        self._image_pipeline = image_pipeline or ImagePipeline(settings.inference)
        self._workspace_manager = workspace_manager or WorkspaceManager(settings.runtime.work_dir)
        self._worker_id = settings.runtime.worker_id or f'worker-{os.getpid()}'
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel = None

    def start(self) -> None:
        while True:
            try:
                self._ensure_channel()
                logger.info(
                    'consumer_start %s',
                    format_log_fields({
                        'workerId': self._worker_id,
                        'queue': self._mq.task_queue,
                        'prefetch': self._mq.prefetch,
                    }),
                )
                self._channel.basic_consume(
                    queue=self._mq.task_queue,
                    on_message_callback=self._on_message,
                    auto_ack=False,
                )
                self._channel.start_consuming()
            except KeyboardInterrupt:
                raise
            except (pika_exceptions.AMQPError, OSError) as exc:
                logger.warning(
                    'consumer_reconnect %s',
                    format_log_fields({
                        'workerId': self._worker_id,
                        'queue': self._mq.task_queue,
                        'error': str(exc),
                    }),
                )
                self._reset_connection()
                time.sleep(2)

    def consume_once(self, timeout_seconds: int = 30) -> bool:
        self._ensure_channel()
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            method, properties, body = self._channel.basic_get(queue=self._mq.task_queue, auto_ack=False)
            if method is None:
                time.sleep(0.2)
                continue
            self._on_message(self._channel, method, properties, body)
            return True
        return False

    def _ensure_channel(self) -> None:
        if (
            self._channel is not None
            and getattr(self._channel, 'is_open', False)
            and self._connection is not None
            and self._connection.is_open
        ):
            return
        # 每次连接/重连都声明拓扑，保证冷启动和断线恢复可用。
        self._reset_connection()
        self._connection = pika.BlockingConnection(pika.URLParameters(self._mq.url))
        self._channel = self._connection.channel()
        self._channel.basic_qos(prefetch_count=self._mq.prefetch)
        self._declare_topology()
        logger.info(
            'mq_connected %s',
            format_log_fields({
                'workerId': self._worker_id,
                'taskQueue': self._mq.task_queue,
                'taskExchange': self._mq.task_exchange,
                'retryExchange': self._mq.retry_exchange,
                'prefetch': self._mq.prefetch,
            }),
        )

    def _reset_connection(self) -> None:
        if self._channel is not None:
            try:
                if self._channel.is_open:
                    self._channel.close()
            except Exception:
                pass
            finally:
                self._channel = None

        if self._connection is not None:
            try:
                if self._connection.is_open:
                    self._connection.close()
            except Exception:
                pass
            finally:
                self._connection = None

    def _declare_topology(self) -> None:
        self._channel.exchange_declare(
            exchange=self._mq.task_exchange,
            exchange_type='direct',
            durable=True,
        )
        self._channel.exchange_declare(
            exchange=self._mq.retry_exchange,
            exchange_type='direct',
            durable=True,
        )
        self._channel.queue_declare(queue=self._mq.task_queue, durable=True)
        self._channel.queue_bind(
            queue=self._mq.task_queue,
            exchange=self._mq.task_exchange,
            routing_key=self._mq.task_routing_key,
        )

        for delay in self.RETRY_DELAYS_SECONDS:
            retry_queue = f'{self._mq.task_queue}.retry.{delay}s'
            retry_routing_key = f'retry.{delay}s'
            self._channel.queue_declare(
                queue=retry_queue,
                durable=True,
                arguments={
                    'x-message-ttl': delay * 1000,
                    'x-dead-letter-exchange': self._mq.task_exchange,
                    'x-dead-letter-routing-key': self._mq.task_routing_key,
                },
            )
            self._channel.queue_bind(
                queue=retry_queue,
                exchange=self._mq.retry_exchange,
                routing_key=retry_routing_key,
            )

    def _on_message(self, channel, method, properties, body) -> None:
        task: Optional[TaskMessage] = None
        workspace = None
        started_at = time.time()

        logger.info(
            'task_message_received %s',
            format_log_fields({
                'workerId': self._worker_id,
                'phase': 'consume',
                'deliveryTag': getattr(method, 'delivery_tag', None),
                'bodyBytes': len(body),
            }),
        )

        try:
            payload = json.loads(body.decode('utf-8'))
            task = TaskMessage.from_dict(payload)
            logger.info('task_schema_validated %s', self._task_log(task, phase='schema_validate'))

            if task.task_type != 'image':
                raise ServiceError(
                    code=ErrorCode.TYPE_NOT_SUPPORTED,
                    message=f'Unsupported task type: {task.task_type}',
                    retryable=False,
                )

            if self._idempotency_store.is_processed(task.event_id):
                logger.info('task_duplicate_skipped %s', self._task_log(task, phase='idempotency'))
                # 重复投递直接 ack，避免重复推理。
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            self._save_event(task, 'RECEIVED', payload_json=payload)
            self._publisher.publish_result(_build_result_payload(task, status='RUNNING', progress=5))
            logger.info('task_running_published %s', self._task_log(task, phase='running', status='RUNNING', progress=5))

            # 每个任务独立工作目录，便于隔离与清理。
            workspace = self._workspace_manager.create(task.task_id, task.attempt)
            ext = os.path.splitext(task.input_file_key)[1] or '.png'
            local_input = os.path.join(workspace.input_path, f'input{ext}')
            local_output = os.path.join(workspace.output_path, f'output{ext}')
            logger.info(
                'task_workspace_ready %s',
                self._task_log(
                    task,
                    phase='workspace',
                    workspace=workspace.task_root,
                    inputLocalPath=local_input,
                    outputLocalPath=local_output,
                ),
            )

            # 核心主链路：下载 -> 推理 -> 上传。
            download_started = time.time()
            logger.info('task_phase_start %s', self._task_log(task, phase='download'))
            self._storage.download(task.input_file_key, local_input)
            download_cost_ms = int((time.time() - download_started) * 1000)
            self._save_event(task, 'DOWNLOADED', payload_json={'inputLocalPath': local_input})
            logger.info(
                'task_phase_done %s',
                self._task_log(task, phase='download', status='DONE', costMs=download_cost_ms),
            )

            infer_started = time.time()
            logger.info('task_phase_start %s', self._task_log(task, phase='enhance'))
            infer_result = self._image_pipeline.run(local_input, local_output, task.scale)
            infer_cost_ms = int((time.time() - infer_started) * 1000)
            self._save_event(task, 'INFERRED', payload_json={'outputLocalPath': infer_result.output_path})
            logger.info(
                'task_phase_done %s',
                self._task_log(
                    task,
                    phase='enhance',
                    status='DONE',
                    costMs=max(infer_cost_ms, infer_result.cost_ms),
                    outputLocalPath=infer_result.output_path,
                ),
            )

            output_key = _build_output_key(task, ext)
            upload_started = time.time()
            logger.info('task_phase_start %s', self._task_log(task, phase='upload', outputFileKey=output_key))
            self._storage.upload(infer_result.output_path, output_key)
            upload_cost_ms = int((time.time() - upload_started) * 1000)
            self._save_event(task, 'UPLOADED', payload_json={'outputFileKey': output_key})
            logger.info(
                'task_phase_done %s',
                self._task_log(task, phase='upload', status='DONE', costMs=upload_cost_ms, outputFileKey=output_key),
            )

            total_cost_ms = int((time.time() - started_at) * 1000)
            self._publisher.publish_result(
                _build_result_payload(
                    task,
                    status='SUCCEEDED',
                    progress=100,
                    output_file_key=output_key,
                    cost_ms=max(total_cost_ms, infer_result.cost_ms),
                ),
            )
            self._save_event(task, 'SUCCEEDED', payload_json={'outputFileKey': output_key, 'costMs': total_cost_ms})
            # 仅在终态后标记幂等，避免重试消息被提前去重。
            self._idempotency_store.mark_processed(task.event_id)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(
                'task_succeeded %s',
                self._task_log(
                    task,
                    phase='done',
                    status='SUCCEEDED',
                    costMs=total_cost_ms,
                    outputFileKey=output_key,
                ),
            )
        except ServiceError as exc:
            self._handle_service_error(channel, method, properties, body, task, exc)
        except Exception as exc:
            self._handle_service_error(
                channel,
                method,
                properties,
                body,
                task,
                ServiceError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=str(exc),
                    retryable=True,
                    cause=exc,
                ),
            )
        finally:
            if workspace is not None:
                self._workspace_manager.cleanup(workspace)
                if task is not None:
                    logger.info(
                        'task_workspace_cleaned %s',
                        self._task_log(task, phase='cleanup', workspace=workspace.task_root),
                    )

    def _handle_service_error(self, channel, method, properties, body, task: Optional[TaskMessage], error: ServiceError) -> None:
        retry_attempt = _get_retry_attempt(properties)
        can_retry = error.retryable and retry_attempt < len(self.RETRY_DELAYS_SECONDS)
        if can_retry:
            # 保留原始消息体，并在 header 中递增重试次数。
            delay = self.RETRY_DELAYS_SECONDS[retry_attempt]
            try:
                self._ensure_channel()
                self._channel.basic_publish(
                    exchange=self._mq.retry_exchange,
                    routing_key=f'retry.{delay}s',
                    body=body,
                    properties=pika.BasicProperties(
                        content_type='application/json',
                        delivery_mode=2,
                        headers={'x-retry-attempt': retry_attempt + 1},
                    ),
                )
                channel.basic_ack(delivery_tag=method.delivery_tag)
                retry_fields = {
                    'phase': 'retry',
                    'status': 'RETRYING',
                    'errorCode': error.code.value,
                    'errorMsg': str(error),
                    'retryAttempt': retry_attempt + 1,
                    'retryDelaySec': delay,
                    'retryable': error.retryable,
                }
                if task is not None:
                    logger.warning('task_retry_routed %s', self._task_log(task, **retry_fields))
                else:
                    retry_fields['workerId'] = self._worker_id
                    logger.warning('task_retry_routed %s', format_log_fields(retry_fields))
            except (pika_exceptions.AMQPError, OSError) as exc:
                logger.warning(
                    'task_retry_route_failed %s',
                    format_log_fields({
                        'workerId': self._worker_id,
                        'errorCode': error.code.value,
                        'errorMsg': str(error),
                        'retryAttempt': retry_attempt + 1,
                        'retryDelaySec': delay,
                        'cause': str(exc),
                    }),
                )
                try:
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                except Exception:
                    pass
            return

        if task is not None:
            self._publisher.publish_result(
                _build_result_payload(
                    task,
                    status='FAILED',
                    progress=100,
                    error_code=error.code.value,
                    error_msg=str(error),
                ),
            )
            self._save_event(task, 'FAILED', error_code=error.code.value, error_msg=str(error))
            # 失败终态同样标记幂等，避免消息无限回投。
            self._idempotency_store.mark_processed(task.event_id)
            logger.error(
                'task_failed %s',
                self._task_log(
                    task,
                    phase='done',
                    status='FAILED',
                    errorCode=error.code.value,
                    errorMsg=str(error),
                    retryable=error.retryable,
                ),
            )
        else:
            fallback_payload = _parse_fallback_payload(body)
            self._publisher.publish_result(fallback_payload)
            logger.error(
                'task_failed_schema %s',
                format_log_fields(
                    {
                        'workerId': self._worker_id,
                        'phase': 'schema_validate',
                        'status': 'FAILED',
                        'errorCode': ErrorCode.SCHEMA_INVALID.value,
                        'errorMsg': 'Invalid task message payload',
                    },
                ),
            )

        channel.basic_ack(delivery_tag=method.delivery_tag)

    def _save_event(
        self,
        task: TaskMessage,
        event_type: str,
        payload_json: Optional[Dict[str, Any]] = None,
        error_code: str = '',
        error_msg: str = '',
    ) -> None:
        try:
            # 事件落库采用尽力而为策略，避免数据库抖动阻塞主流程。
            self._event_repo.save_event(TaskEventRecord(
                task_id=task.task_id,
                task_no=task.task_no,
                event_type=event_type,
                event_time=datetime.now(timezone.utc),
                attempt=task.attempt,
                worker_id=self._worker_id,
                payload_json=payload_json,
                error_code=error_code,
                error_msg=error_msg,
                trace_id=task.trace_id,
            ))
            logger.info('task_event_saved %s', self._task_log(task, phase='event', eventType=event_type))
        except ServiceError as exc:
            # 即使事件写库失败，也保持主流程继续执行。
            logger.warning(
                'task_event_save_failed %s',
                self._task_log(
                    task,
                    phase='event',
                    status='FAILED',
                    eventType=event_type,
                    errorCode=exc.code.value,
                    errorMsg=str(exc),
                ),
            )

    def _task_log(self, task: TaskMessage, **extra: Any) -> str:
        fields: Dict[str, Any] = {
            'workerId': self._worker_id,
            'traceId': task.trace_id,
            'taskId': task.task_id,
            'taskNo': task.task_no,
            'eventId': task.event_id,
            'attempt': task.attempt,
        }
        fields.update(extra)
        return format_log_fields(fields)


def _build_output_key(task: TaskMessage, extension: str) -> str:
    clean_ext = extension if extension.startswith('.') else f'.{extension}'
    return f'output/{task.task_id}/{task.task_no}_x{task.scale}{clean_ext}'


def _build_result_payload(
    task: TaskMessage,
    status: str,
    progress: int,
    output_file_key: Optional[str] = None,
    cost_ms: int = 0,
    error_code: Optional[str] = None,
    error_msg: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        'schemaVersion': task.schema_version,
        'eventId': f'evt_result_{uuid4().hex}',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'taskId': task.task_id,
        'status': status,
        'progress': progress,
        'outputFileKey': output_file_key,
        'costMs': cost_ms,
        'attempt': task.attempt,
        'errorCode': error_code,
        'errorMsg': error_msg,
        'traceId': task.trace_id,
    }


def _get_retry_attempt(properties: Any) -> int:
    headers = getattr(properties, 'headers', None) or {}
    value = headers.get('x-retry-attempt', 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_fallback_payload(body: bytes) -> Dict[str, Any]:
    trace_id = ''
    task_id = 0
    attempt = 1
    try:
        payload = json.loads(body.decode('utf-8'))
        trace_id = str(payload.get('traceId', ''))
        task_id = int(payload.get('taskId', 0) or 0)
        attempt = int(payload.get('attempt', 1) or 1)
    except Exception:
        pass

    return {
        'schemaVersion': '1.0',
        'eventId': f'evt_result_{uuid4().hex}',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'taskId': task_id,
        'status': 'FAILED',
        'progress': 100,
        'outputFileKey': None,
        'costMs': 0,
        'attempt': attempt,
        'errorCode': ErrorCode.SCHEMA_INVALID.value,
        'errorMsg': 'Invalid task message payload',
        'traceId': trace_id,
    }



