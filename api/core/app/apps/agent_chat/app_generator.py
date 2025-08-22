import contextvars
import logging
import threading
import uuid
from collections.abc import Generator, Mapping
from typing import Any, Literal, Union, overload

from flask import Flask, current_app
from pydantic import ValidationError

from configs import dify_config
from constants import UUID_NIL
from core.app.app_config.easy_ui_based_app.model_config.converter import ModelConfigConverter
from core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from core.app.apps.agent_chat.app_config_manager import AgentChatAppConfigManager
from core.app.apps.agent_chat.app_runner import AgentChatAppRunner
from core.app.apps.agent_chat.generate_response_converter import AgentChatAppGenerateResponseConverter
from core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from core.app.apps.exc import GenerateTaskStoppedError
from core.app.apps.message_based_app_generator import MessageBasedAppGenerator
from core.app.apps.message_based_app_queue_manager import MessageBasedAppQueueManager
from core.app.entities.app_invoke_entities import AgentChatAppGenerateEntity, InvokeFrom
from core.model_runtime.errors.invoke import InvokeAuthorizationError
from core.ops.ops_trace_manager import TraceQueueManager
from extensions.ext_database import db
from factories import file_factory
from libs.flask_utils import preserve_flask_contexts
from models import Account, App, EndUser
from services.conversation_service import ConversationService

logger = logging.getLogger(__name__)


class AgentChatAppGenerator(MessageBasedAppGenerator):
    @overload
    def generate(
        self,
        *,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: Literal[False],
    ) -> Mapping[str, Any]: ...

    @overload
    def generate(
        self,
        *,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: Literal[True],
    ) -> Generator[Mapping | str, None, None]: ...

    @overload
    def generate(
        self,
        *,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool,
    ) -> Union[Mapping, Generator[Mapping | str, None, None]]: ...

    def generate(
        self,
        *,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool = True,
    ) -> Union[Mapping, Generator[Mapping | str, None, None]]:
        """
        Generate App response.

        :param app_model: App
        :param user: account or end user
        :param args: request args
        :param invoke_from: invoke from source
        :param streaming: is stream
        """
        # Agent模式强制流式响应，保证实时交互体验
        if not streaming:
            raise ValueError("Agent Chat App does not support blocking mode")

        # 验证必要参数
        if not args.get("query"):
            raise ValueError("query is required")

        query = args["query"]
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        # 清理query中的空字符，NULL字符
        query = query.replace("\x00", "")
        inputs = args["inputs"]

        extras = {"auto_generate_conversation_name": args.get("auto_generate_name", True)}

        # 获取或创建对话实例
        conversation = None
        conversation_id = args.get("conversation_id")
        if conversation_id:
            conversation = ConversationService.get_conversation(
                app_model=app_model, conversation_id=conversation_id, user=user
            )
            
        # 获取应用模型配置
        app_model_config = self._get_app_model_config(app_model=app_model, conversation=conversation)

        # 处理调试模式的配置覆盖
        override_model_config_dict = None
        if args.get("model_config"):
            if invoke_from != InvokeFrom.DEBUGGER:
                raise ValueError("Only in App debug mode can override model config")

            # 验证覆盖配置的有效性
            override_model_config_dict = AgentChatAppConfigManager.config_validate(
                tenant_id=app_model.tenant_id,
                config=args["model_config"],
            )

            # 调试模式下总是启用检索资源
            override_model_config_dict["retriever_resource"] = {"enabled": True}

        # 解析文件上传（支持多模态输入）
        # TODO(QuantumGhost): Move file parsing logic to the API controller layer
        # for better separation of concerns.
        #
        # For implementation reference, see the `_parse_file` function and
        # `DraftWorkflowNodeRunApi` class which handle this properly.
        files = args.get("files") or []
        file_extra_config = FileUploadConfigManager.convert(override_model_config_dict or app_model_config.to_dict())
        if file_extra_config:
            # 根据配置构建文件对象
            file_objs = file_factory.build_from_mappings(
                mappings=files,
                tenant_id=app_model.tenant_id,
                config=file_extra_config,
            )
        else:
            file_objs = []

        # 转换为应用配置对象
        app_config = AgentChatAppConfigManager.get_app_config(
            app_model=app_model,
            app_model_config=app_model_config,
            conversation=conversation,
            override_config_dict=override_model_config_dict,
        )

        # 初始化追踪管理器（用于性能监控和调试）
        trace_manager = TraceQueueManager(app_model.id, user.id if isinstance(user, Account) else user.session_id)

        # 创建应用生成实体，包含完整的执行上下文
        application_generate_entity = AgentChatAppGenerateEntity(
            task_id=str(uuid.uuid4()),              # 全局唯一任务ID
            app_config=app_config,                  # 应用配置
            model_conf=ModelConfigConverter.convert(app_config),  # 模型配置
            file_upload_config=file_extra_config,   # 文件上传配置
            conversation_id=conversation.id if conversation else None,
            inputs=self._prepare_user_inputs(       # 处理用户输入变量
                user_inputs=inputs, variables=app_config.variables, tenant_id=app_model.tenant_id
            ),
            query=query,                            # 用户查询
            files=list(file_objs),                  # 文件对象列表
            parent_message_id=args.get("parent_message_id") if invoke_from != InvokeFrom.SERVICE_API else UUID_NIL,
            user_id=user.id,
            stream=streaming,
            invoke_from=invoke_from,
            extras=extras,
            call_depth=0,                           # 递归调用深度
            trace_manager=trace_manager,
        )

        # 初始化数据库记录（conversation和message）
        (conversation, message) = self._init_generate_records(application_generate_entity, conversation)

        # 初始化队列管理器，用于实时事件流传输
        queue_manager = MessageBasedAppQueueManager(
            task_id=application_generate_entity.task_id,
            user_id=application_generate_entity.user_id,
            invoke_from=application_generate_entity.invoke_from,
            conversation_id=conversation.id,
            app_mode=conversation.mode,
            message_id=message.id,
        )

        # 复制当前上下文变量，确保线程间上下文隔离
        context = contextvars.copy_context()

        # 创建工作线程执行智能体推理
        worker_thread = threading.Thread(
            target=self._generate_worker,
            kwargs={
                "flask_app": current_app._get_current_object(),  # Flask应用实例
                "context": context,                              # 上下文变量
                "application_generate_entity": application_generate_entity,
                "queue_manager": queue_manager,                  # 队列管理器
                "conversation_id": conversation.id,
                "message_id": message.id,
            },
        )

        # 启动异步执行线程
        worker_thread.start()

        # 处理响应并返回流式生成器
        response = self._handle_response(
            application_generate_entity=application_generate_entity,
            queue_manager=queue_manager,
            conversation=conversation,
            message=message,
            user=user,
            stream=streaming,
        )
        
        # 转换响应格式并返回
        return AgentChatAppGenerateResponseConverter.convert(response=response, invoke_from=invoke_from)  # type: ignore

    def _generate_worker(
        self,
        flask_app: Flask,
        context: contextvars.Context,
        application_generate_entity: AgentChatAppGenerateEntity,
        queue_manager: AppQueueManager,
        conversation_id: str,
        message_id: str,
    ) -> None:
        """
        Generate worker in a new thread.
        :param flask_app: Flask app
        :param application_generate_entity: application generate entity
        :param queue_manager: queue manager
        :param conversation_id: conversation ID
        :param message_id: message ID
        :return:
        """

        # 保持Flask上下文和上下文变量
        with preserve_flask_contexts(flask_app, context_vars=context):
            try:
                # 重新获取对话和消息实例（跨线程数据库访问）
                conversation = self._get_conversation(conversation_id)
                message = self._get_message(message_id)

                # 创建并运行智能体执行器
                runner = AgentChatAppRunner()
                runner.run(
                    application_generate_entity=application_generate_entity,
                    queue_manager=queue_manager,              # 通过队列发布实时事件
                    conversation=conversation,
                    message=message,
                )
            except GenerateTaskStoppedError:
                # 任务被用户手动停止，正常结束
                pass
            except InvokeAuthorizationError:
                # API密钥错误，发布授权错误事件
                queue_manager.publish_error(
                    InvokeAuthorizationError("Incorrect API key provided"), PublishFrom.APPLICATION_MANAGER
                )
            except ValidationError as e:
                # 配置验证失败
                logger.exception("Validation Error when generating")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            except ValueError as e:
                # 参数错误
                if dify_config.DEBUG:
                    logger.exception("Error when generating")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            except Exception as e:
                # 未知错误，记录详细日志
                logger.exception("Unknown Error when generating")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            finally:
                # 确保数据库连接正确关闭
                db.session.close()
