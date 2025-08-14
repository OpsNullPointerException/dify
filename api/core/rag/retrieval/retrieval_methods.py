from enum import Enum


class RetrievalMethod(Enum):
    """
    检索方法枚举类
    定义了不同的文档检索策略
    """
    SEMANTIC_SEARCH = "semantic_search"      # 语义搜索：基于向量相似度的智能检索
    FULL_TEXT_SEARCH = "full_text_search"    # 全文搜索：基于关键词匹配的传统检索
    HYBRID_SEARCH = "hybrid_search"          # 混合搜索：结合语义搜索和全文搜索

    @staticmethod
    def is_support_semantic_search(retrieval_method: str) -> bool:
        """检查检索方法是否支持语义搜索"""
        return retrieval_method in {RetrievalMethod.SEMANTIC_SEARCH.value, RetrievalMethod.HYBRID_SEARCH.value}

    @staticmethod
    def is_support_fulltext_search(retrieval_method: str) -> bool:
        """检查检索方法是否支持全文搜索"""
        return retrieval_method in {RetrievalMethod.FULL_TEXT_SEARCH.value, RetrievalMethod.HYBRID_SEARCH.value}
