from typing import Any, Optional

from openai import BaseModel
from pydantic import Field

from core.app.entities.app_invoke_entities import InvokeFrom
from core.tools.entities.tool_entities import CredentialType, ToolInvokeFrom


class ToolRuntime(BaseModel):
    """
    工具运行时上下文 - 工具执行时的元数据和环境信息
    
    包含工具执行所需的所有上下文信息：
    1. 租户信息：tenant_id 标识工具所属的租户
    2. 调用来源：区分来自Agent、Workflow、调试器等不同场景的调用
    3. 认证凭据：存储工具访问外部服务所需的API密钥等认证信息
    4. 运行时参数：工具执行时的动态配置参数
    5. 凭据类型：支持API_KEY、OAuth等不同的认证方式
    
    这个类确保了工具在不同执行环境中都能获得正确的上下文信息
    """

    tenant_id: str
    tool_id: Optional[str] = None
    invoke_from: Optional[InvokeFrom] = None
    tool_invoke_from: Optional[ToolInvokeFrom] = None
    credentials: dict[str, Any] = Field(default_factory=dict)
    credential_type: CredentialType = Field(default=CredentialType.API_KEY)
    runtime_parameters: dict[str, Any] = Field(default_factory=dict)


class FakeToolRuntime(ToolRuntime):
    """
    Fake tool runtime for testing
    """

    def __init__(self):
        super().__init__(
            tenant_id="fake_tenant_id",
            tool_id="fake_tool_id",
            invoke_from=InvokeFrom.DEBUGGER,
            tool_invoke_from=ToolInvokeFrom.AGENT,
            credentials={},
            runtime_parameters={},
        )
