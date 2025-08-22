import logging
import time
import uuid
from collections.abc import Generator, Mapping
from datetime import timedelta
from typing import Any, Optional, Union

from core.errors.error import AppInvokeQuotaExceededError
from extensions.ext_redis import redis_client

logger = logging.getLogger(__name__)


class RateLimit:
    """并发请求限流器 - 基于Redis哈希表实现的分布式并发控制"""
    _MAX_ACTIVE_REQUESTS_KEY = "dify:rate_limit:{}:max_active_requests"  # 存储最大并发数配置
    _ACTIVE_REQUESTS_KEY = "dify:rate_limit:{}:active_requests"          # 存储当前活跃请求列表
    _UNLIMITED_REQUEST_ID = "unlimited_request_id"                       # 无限制请求的标识符
    _REQUEST_MAX_ALIVE_TIME = 10 * 60  # 请求最大存活时间：10分钟（防止僵尸请求）
    _ACTIVE_REQUESTS_COUNT_FLUSH_INTERVAL = 5 * 60  # 每5分钟重新统计活跃请求数（清理过期请求）
    _instance_dict: dict[str, "RateLimit"] = {}  # 单例模式：每个client_id对应一个RateLimit实例

    def __new__(cls: type["RateLimit"], client_id: str, max_active_requests: int):
        """单例模式：同一个client_id只创建一个RateLimit实例，避免重复计数"""
        if client_id not in cls._instance_dict:
            instance = super().__new__(cls)
            cls._instance_dict[client_id] = instance
        return cls._instance_dict[client_id]

    def __init__(self, client_id: str, max_active_requests: int):
        self.max_active_requests = max_active_requests
        # must be called after max_active_requests is set
        if self.disabled():
            return
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        self.client_id = client_id
        self.active_requests_key = self._ACTIVE_REQUESTS_KEY.format(client_id)
        self.max_active_requests_key = self._MAX_ACTIVE_REQUESTS_KEY.format(client_id)
        self.last_recalculate_time = float("-inf")
        self.flush_cache(use_local_value=True)

    def flush_cache(self, use_local_value=False):
        if self.disabled():
            return
        self.last_recalculate_time = time.time()
        # flush max active requests
        if use_local_value or not redis_client.exists(self.max_active_requests_key):
            redis_client.setex(self.max_active_requests_key, timedelta(days=1), self.max_active_requests)
        else:
            self.max_active_requests = int(redis_client.get(self.max_active_requests_key).decode("utf-8"))
            redis_client.expire(self.max_active_requests_key, timedelta(days=1))

        # flush max active requests (in-transit request list)
        if not redis_client.exists(self.active_requests_key):
            return
        request_details = redis_client.hgetall(self.active_requests_key)
        redis_client.expire(self.active_requests_key, timedelta(days=1))
        timeout_requests = [
            k
            for k, v in request_details.items()
            if time.time() - float(v.decode("utf-8")) > RateLimit._REQUEST_MAX_ALIVE_TIME
        ]
        if timeout_requests:
            redis_client.hdel(self.active_requests_key, *timeout_requests)

    def enter(self, request_id: Optional[str] = None) -> str:
        """请求进入限流器 - 检查并发数限制，注册活跃请求"""
        if self.disabled():
            return RateLimit._UNLIMITED_REQUEST_ID
            
        # 定期清理过期请求（每5分钟）
        if time.time() - self.last_recalculate_time > RateLimit._ACTIVE_REQUESTS_COUNT_FLUSH_INTERVAL:
            self.flush_cache()
            
        if not request_id:
            request_id = RateLimit.gen_request_key()

        # 检查当前活跃请求数：使用Redis哈希表长度作为并发计数器
        active_requests_count = redis_client.hlen(self.active_requests_key)
        if active_requests_count >= self.max_active_requests:
            raise AppInvokeQuotaExceededError(
                f"Too many requests. Please try again later. The current maximum concurrent requests allowed "
                f"for {self.client_id} is {self.max_active_requests}."
            )
        
        # 将请求ID注册到活跃请求哈希表：{request_id: timestamp}
        redis_client.hset(self.active_requests_key, request_id, str(time.time()))
        return request_id

    def exit(self, request_id: str):
        """请求退出限流器 - 从活跃请求列表中移除，释放并发槽位"""
        if request_id == RateLimit._UNLIMITED_REQUEST_ID:
            return
        # 从Redis哈希表中删除请求记录，释放一个并发槽位
        redis_client.hdel(self.active_requests_key, request_id)

    def disabled(self):
        return self.max_active_requests <= 0

    @staticmethod
    def gen_request_key() -> str:
        """生成全局唯一的请求ID - 使用UUID4确保跨进程、跨时间的唯一性"""
        return str(uuid.uuid4())  # 格式: "550e8400-e29b-41d4-a716-446655440000"

    def generate(self, generator: Union[Generator[str, None, None], Mapping[str, Any]], request_id: str):
        """包装Generator以实现自动限流管理"""
        if isinstance(generator, Mapping):
            # 非流式响应：直接返回数据，无需限流包装
            return generator
        else:
            # 流式响应：用RateLimitGenerator包装，实现自动进入/退出限流
            return RateLimitGenerator(rate_limit=self, generator=generator, request_id=request_id)


class RateLimitGenerator:
    """带限流功能的生成器包装器 - 在生成器结束时自动释放并发槽位"""
    def __init__(self, rate_limit: RateLimit, generator: Generator[str, None, None], request_id: str):
        self.rate_limit = rate_limit
        self.generator = generator
        self.request_id = request_id
        self.closed = False  # 防止重复关闭

    def __iter__(self):
        return self

    def __next__(self):
        """迭代器协议 - 获取下一个数据，异常时自动清理限流状态"""
        if self.closed:
            raise StopIteration
        try:
            return next(self.generator)  # 从原始生成器获取数据
        except Exception:
            self.close()  # 发生任何异常时释放并发槽位
            raise

    def close(self):
        """关闭生成器并释放限流资源 - 确保并发槽位被正确释放"""
        if not self.closed:
            self.closed = True
            # 从限流器中退出，释放并发槽位
            self.rate_limit.exit(self.request_id)
            # 关闭原始生成器（如果支持）
            if self.generator is not None and hasattr(self.generator, "close"):
                self.generator.close()
