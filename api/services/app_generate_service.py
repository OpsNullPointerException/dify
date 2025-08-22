import uuid
from collections.abc import Generator, Mapping
from typing import Any, Optional, Union

from openai._exceptions import RateLimitError

from configs import dify_config
from core.app.apps.advanced_chat.app_generator import AdvancedChatAppGenerator
from core.app.apps.agent_chat.app_generator import AgentChatAppGenerator
from core.app.apps.chat.app_generator import ChatAppGenerator
from core.app.apps.completion.app_generator import CompletionAppGenerator
from core.app.apps.workflow.app_generator import WorkflowAppGenerator
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.features.rate_limiting import RateLimit
from libs.helper import RateLimiter
from models.model import Account, App, AppMode, EndUser
from models.workflow import Workflow
from services.billing_service import BillingService
from services.errors.app import WorkflowIdFormatError, WorkflowNotFoundError
from services.errors.llm import InvokeRateLimitError
from services.workflow_service import WorkflowService


class AppGenerateService:
    # 系统级日频率限制器：基于Redis Sorted Set实现的滑动窗口限流
    # 参数：前缀，最大请求数/天，时间窗口(86400秒=24小时)
    system_rate_limiter = RateLimiter("app_daily_rate_limiter", dify_config.APP_DAILY_RATE_LIMIT, 86400)

    @classmethod
    def generate(
        cls,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool = True,
    ):
        """
        App Content Generate
        :param app_model: app model
        :param user: user
        :param args: args
        :param invoke_from: invoke from
        :param streaming: streaming
        :return:
        """
        # 系统级限流：基于租户的日请求次数限制（仅针对免费用户）
        if dify_config.BILLING_ENABLED:
            # 检查是否为免费套餐
            limit_info = BillingService.get_info(app_model.tenant_id)
            if limit_info["subscription"]["plan"] == "sandbox":
                # 检查日频率限制：使用tenant_id作为限流维度
                if cls.system_rate_limiter.is_rate_limited(app_model.tenant_id):
                    raise InvokeRateLimitError(
                        "Rate limit exceeded, please upgrade your plan "
                        f"or your RPD was {dify_config.APP_DAILY_RATE_LIMIT} requests/day"
                    )
                # 增加请求计数：在Redis Sorted Set中记录当前时间戳
                cls.system_rate_limiter.increment_rate_limit(app_model.tenant_id)

        # 应用级限流：基于应用的并发请求数限制
        max_active_request = AppGenerateService._get_max_active_requests(app_model)
        rate_limit = RateLimit(app_model.id, max_active_request)  # client_id = app.id
        request_id = RateLimit.gen_request_key()
        try:
            # 进入并发限流器，获取实际的请求ID
            request_id = rate_limit.enter(request_id)
            
            # 根据应用模式分发到不同的生成器
            if app_model.mode == AppMode.COMPLETION.value:
                # COMPLETION模式：文本生成应用（一问一答，无对话历史）
                # 适用场景：文本摘要、翻译、内容生成等
                return rate_limit.generate(
                    CompletionAppGenerator.convert_to_event_stream(
                        CompletionAppGenerator().generate(
                            app_model=app_model, user=user, args=args, invoke_from=invoke_from, streaming=streaming
                        ),
                    ),
                    request_id=request_id,
                )
            elif app_model.mode == AppMode.AGENT_CHAT.value or app_model.is_agent:
                # AGENT_CHAT模式：智能体对话应用（支持Function Call工具调用）
                # 适用场景：智能助手、任务执行、API调用等
                return rate_limit.generate(
                    AgentChatAppGenerator.convert_to_event_stream(
                        AgentChatAppGenerator().generate(
                            app_model=app_model, user=user, args=args, invoke_from=invoke_from, streaming=streaming
                        ),
                    ),
                    request_id,
                )
            elif app_model.mode == AppMode.CHAT.value:
                # CHAT模式：基础对话应用（简单配置，有对话历史）
                # 适用场景：客服机器人、简单问答系统等
                return rate_limit.generate(
                    ChatAppGenerator.convert_to_event_stream(
                        ChatAppGenerator().generate(
                            app_model=app_model, user=user, args=args, invoke_from=invoke_from, streaming=streaming
                        ),
                    ),
                    request_id=request_id,
                )
            elif app_model.mode == AppMode.ADVANCED_CHAT.value:
                # ADVANCED_CHAT模式：基于工作流的高级对话应用
                # 适用场景：复杂业务流程、多步骤交互、可视化配置等
                workflow_id = args.get("workflow_id")
                workflow = cls._get_workflow(app_model, invoke_from, workflow_id)
                return rate_limit.generate(
                    AdvancedChatAppGenerator.convert_to_event_stream(
                        AdvancedChatAppGenerator().generate(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=streaming,
                        ),
                    ),
                    request_id=request_id,
                )
            elif app_model.mode == AppMode.WORKFLOW.value:
                # WORKFLOW模式：纯工作流执行应用（无对话界面）
                # 适用场景：数据处理、自动化任务、批处理流水线等
                workflow_id = args.get("workflow_id")
                workflow = cls._get_workflow(app_model, invoke_from, workflow_id)
                return rate_limit.generate(
                    WorkflowAppGenerator.convert_to_event_stream(
                        WorkflowAppGenerator().generate(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=streaming,
                            call_depth=0,
                            workflow_thread_pool_id=None,
                        ),
                    ),
                    request_id,
                )
            else:
                raise ValueError(f"Invalid app mode {app_model.mode}")
        except RateLimitError as e:
            raise InvokeRateLimitError(str(e))
        except Exception:
            rate_limit.exit(request_id)
            raise
        finally:
            if not streaming:
                rate_limit.exit(request_id)

    @staticmethod
    def _get_max_active_requests(app: App) -> int:
        """
        Get the maximum number of active requests allowed for an app.

        Returns the smaller value between app's custom limit and global config limit.
        A value of 0 means infinite (no limit).

        Args:
            app: The App model instance

        Returns:
            The maximum number of active requests allowed
        """
        app_limit = app.max_active_requests or 0
        config_limit = dify_config.APP_MAX_ACTIVE_REQUESTS

        # Filter out infinite (0) values and return the minimum, or 0 if both are infinite
        limits = [limit for limit in [app_limit, config_limit] if limit > 0]
        return min(limits) if limits else 0

    @classmethod
    def generate_single_iteration(cls, app_model: App, user: Account, node_id: str, args: Any, streaming: bool = True):
        if app_model.mode == AppMode.ADVANCED_CHAT.value:
            workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER)
            return AdvancedChatAppGenerator.convert_to_event_stream(
                AdvancedChatAppGenerator().single_iteration_generate(
                    app_model=app_model, workflow=workflow, node_id=node_id, user=user, args=args, streaming=streaming
                )
            )
        elif app_model.mode == AppMode.WORKFLOW.value:
            workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER)
            return AdvancedChatAppGenerator.convert_to_event_stream(
                WorkflowAppGenerator().single_iteration_generate(
                    app_model=app_model, workflow=workflow, node_id=node_id, user=user, args=args, streaming=streaming
                )
            )
        else:
            raise ValueError(f"Invalid app mode {app_model.mode}")

    @classmethod
    def generate_single_loop(cls, app_model: App, user: Account, node_id: str, args: Any, streaming: bool = True):
        if app_model.mode == AppMode.ADVANCED_CHAT.value:
            workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER)
            return AdvancedChatAppGenerator.convert_to_event_stream(
                AdvancedChatAppGenerator().single_loop_generate(
                    app_model=app_model, workflow=workflow, node_id=node_id, user=user, args=args, streaming=streaming
                )
            )
        elif app_model.mode == AppMode.WORKFLOW.value:
            workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER)
            return AdvancedChatAppGenerator.convert_to_event_stream(
                WorkflowAppGenerator().single_loop_generate(
                    app_model=app_model, workflow=workflow, node_id=node_id, user=user, args=args, streaming=streaming
                )
            )
        else:
            raise ValueError(f"Invalid app mode {app_model.mode}")

    @classmethod
    def generate_more_like_this(
        cls,
        app_model: App,
        user: Union[Account, EndUser],
        message_id: str,
        invoke_from: InvokeFrom,
        streaming: bool = True,
    ) -> Union[Mapping, Generator]:
        """
        Generate more like this
        :param app_model: app model
        :param user: user
        :param message_id: message id
        :param invoke_from: invoke from
        :param streaming: streaming
        :return:
        """
        return CompletionAppGenerator().generate_more_like_this(
            app_model=app_model, message_id=message_id, user=user, invoke_from=invoke_from, stream=streaming
        )

    @classmethod
    def _get_workflow(cls, app_model: App, invoke_from: InvokeFrom, workflow_id: Optional[str] = None) -> Workflow:
        """
        Get workflow
        :param app_model: app model
        :param invoke_from: invoke from
        :param workflow_id: optional workflow id to specify a specific version
        :return:
        """
        workflow_service = WorkflowService()

        # If workflow_id is specified, get the specific workflow version
        if workflow_id:
            try:
                workflow_uuid = uuid.UUID(workflow_id)
            except ValueError:
                raise WorkflowIdFormatError(f"Invalid workflow_id format: '{workflow_id}'. ")
            workflow = workflow_service.get_published_workflow_by_id(app_model=app_model, workflow_id=workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(f"Workflow not found with id: {workflow_id}")
            return workflow

        if invoke_from == InvokeFrom.DEBUGGER:
            # fetch draft workflow by app_model
            workflow = workflow_service.get_draft_workflow(app_model=app_model)

            if not workflow:
                raise ValueError("Workflow not initialized")
        else:
            # fetch published workflow by app_model
            workflow = workflow_service.get_published_workflow(app_model=app_model)

            if not workflow:
                raise ValueError("Workflow not published")

        return workflow
