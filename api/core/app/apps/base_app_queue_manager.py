import queue
import time
from abc import abstractmethod
from enum import Enum
from typing import Any, Optional

from sqlalchemy.orm import DeclarativeMeta

from configs import dify_config
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.entities.queue_entities import (
    AppQueueEvent,
    MessageQueueMessage,
    QueueErrorEvent,
    QueuePingEvent,
    QueueStopEvent,
    WorkflowQueueMessage,
)
from extensions.ext_redis import redis_client


class PublishFrom(Enum):
    APPLICATION_MANAGER = 1
    TASK_PIPELINE = 2


class AppQueueManager:
    """应用队列管理器基类，使用内存队列实现事件流式传输"""
    
    def __init__(self, task_id: str, user_id: str, invoke_from: InvokeFrom) -> None:
        if not user_id:
            raise ValueError("user is required")

        self._task_id = task_id
        self._user_id = user_id
        self._invoke_from = invoke_from

        # Redis缓存任务归属信息，30分钟过期
        user_prefix = "account" if self._invoke_from in {InvokeFrom.EXPLORE, InvokeFrom.DEBUGGER} else "end-user"
        redis_client.setex(
            AppQueueManager._generate_task_belong_cache_key(self._task_id), 1800, f"{user_prefix}-{self._user_id}"
        )

        # 创建内存队列，用于事件流传输
        q: queue.Queue[WorkflowQueueMessage | MessageQueueMessage | None] = queue.Queue()
        self._q = q

    def listen(self):
        """监听队列并生成事件流，供前端SSE消费"""
        listen_timeout = dify_config.APP_MAX_EXECUTION_TIME
        start_time = time.time()
        last_ping_time: int | float = 0
        while True:
            try:
                # 阻塞获取队列消息，1秒超时
                message = self._q.get(timeout=1)
                if message is None:
                    break
                yield message
            except queue.Empty:
                continue
            finally:
                elapsed_time = time.time() - start_time
                # 超时或被停止时发送停止信号
                if elapsed_time >= listen_timeout or self._is_stopped():
                    self.publish(
                        QueueStopEvent(stopped_by=QueueStopEvent.StopBy.USER_MANUAL), PublishFrom.TASK_PIPELINE
                    )

                # 每10秒发送心跳信号
                if elapsed_time // 10 > last_ping_time:
                    self.publish(QueuePingEvent(), PublishFrom.TASK_PIPELINE)
                    last_ping_time = elapsed_time // 10

    def stop_listen(self) -> None:
        """
        Stop listen to queue
        :return:
        """
        self._q.put(None)

    def publish_error(self, e, pub_from: PublishFrom) -> None:
        """
        Publish error
        :param e: error
        :param pub_from: publish from
        :return:
        """
        self.publish(QueueErrorEvent(error=e), pub_from)

    def publish(self, event: AppQueueEvent, pub_from: PublishFrom) -> None:
        """发布事件到队列"""
        self._check_for_sqlalchemy_models(event.model_dump())  # 避免线程安全问题
        self._publish(event, pub_from)

    @abstractmethod
    def _publish(self, event: AppQueueEvent, pub_from: PublishFrom) -> None:
        """
        Publish event to queue
        :param event:
        :param pub_from:
        :return:
        """
        raise NotImplementedError

    @classmethod
    def set_stop_flag(cls, task_id: str, invoke_from: InvokeFrom, user_id: str) -> None:
        """通过Redis设置任务停止标志"""
        result: Optional[Any] = redis_client.get(cls._generate_task_belong_cache_key(task_id))
        if result is None:
            return

        user_prefix = "account" if invoke_from in {InvokeFrom.EXPLORE, InvokeFrom.DEBUGGER} else "end-user"
        if result.decode("utf-8") != f"{user_prefix}-{user_id}":
            return

        stopped_cache_key = cls._generate_stopped_cache_key(task_id)
        redis_client.setex(stopped_cache_key, 600, 1)  # 10分钟过期

    def _is_stopped(self) -> bool:
        """
        Check if task is stopped
        :return:
        """
        stopped_cache_key = AppQueueManager._generate_stopped_cache_key(self._task_id)
        result = redis_client.get(stopped_cache_key)
        if result is not None:
            return True

        return False

    @classmethod
    def _generate_task_belong_cache_key(cls, task_id: str) -> str:
        """
        Generate task belong cache key
        :param task_id: task id
        :return:
        """
        return f"generate_task_belong:{task_id}"

    @classmethod
    def _generate_stopped_cache_key(cls, task_id: str) -> str:
        """
        Generate stopped cache key
        :param task_id: task id
        :return:
        """
        return f"generate_task_stopped:{task_id}"

    def _check_for_sqlalchemy_models(self, data: Any):
        """递归检查数据中是否包含SQLAlchemy模型实例，防止跨线程传递导致的线程安全问题"""
        if isinstance(data, dict):
            for key, value in data.items():
                self._check_for_sqlalchemy_models(value)
        elif isinstance(data, list):
            for item in data:
                self._check_for_sqlalchemy_models(item)
        else:
            # 检测SQLAlchemy模型实例
            if isinstance(data, DeclarativeMeta) or hasattr(data, "_sa_instance_state"):
                raise TypeError(
                    "Critical Error: Passing SQLAlchemy Model instances that cause thread safety issues is not allowed."
                )
