from core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from core.app.apps.exc import GenerateTaskStoppedError
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.entities.queue_entities import (
    AppQueueEvent,
    MessageQueueMessage,
    QueueAdvancedChatMessageEndEvent,
    QueueErrorEvent,
    QueueMessage,
    QueueMessageEndEvent,
    QueueStopEvent,
)


class MessageBasedAppQueueManager(AppQueueManager):
    """基于消息的应用队列管理器，用于Chat/Agent等应用的实时事件流管理"""
    
    def __init__(
        self, task_id: str, user_id: str, invoke_from: InvokeFrom, conversation_id: str, app_mode: str, message_id: str
    ) -> None:
        super().__init__(task_id, user_id, invoke_from)

        self._conversation_id = str(conversation_id)
        self._app_mode = app_mode
        self._message_id = str(message_id)

    def construct_queue_message(self, event: AppQueueEvent) -> QueueMessage:
        return MessageQueueMessage(
            task_id=self._task_id,
            message_id=self._message_id,
            conversation_id=self._conversation_id,
            app_mode=self._app_mode,
            event=event,
        )

    def _publish(self, event: AppQueueEvent, pub_from: PublishFrom) -> None:
        """发布事件到队列供前端实时消费"""
        message = MessageQueueMessage(
            task_id=self._task_id,
            message_id=self._message_id,
            conversation_id=self._conversation_id,
            app_mode=self._app_mode,
            event=event,
        )

        self._q.put(message)

        # 如果是结束类事件，停止监听
        if isinstance(
            event, QueueStopEvent | QueueErrorEvent | QueueMessageEndEvent | QueueAdvancedChatMessageEndEvent
        ):
            self.stop_listen()

        if pub_from == PublishFrom.APPLICATION_MANAGER and self._is_stopped():
            raise GenerateTaskStoppedError()
