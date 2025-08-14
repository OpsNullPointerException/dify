import logging
import os

from core.rag.extractor.extractor_base import BaseExtractor
from core.rag.models.document import Document

logger = logging.getLogger(__name__)


class UnstructuredWordExtractor(BaseExtractor):
    """使用 Unstructured 库加载 Word 文档的提取器"""

    def __init__(self, file_path: str, api_url: str, api_key: str = ""):
        """初始化文件路径和 API 配置"""
        self._file_path = file_path
        self._api_url = api_url
        self._api_key = api_key

    def extract(self) -> list[Document]:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.file_utils.filetype import FileType, detect_filetype

        # 检查 Unstructured 版本兼容性
        unstructured_version = tuple(int(x) for x in __unstructured_version__.split("."))
        
        # 检测文件类型（.doc 还是 .docx）
        try:
            import magic  # noqa: F401
            # 使用 magic 库精确检测文件类型
            is_doc = detect_filetype(self._file_path) == FileType.DOC
        except ImportError:
            # 如果没有 magic 库，根据文件扩展名判断
            _, extension = os.path.splitext(str(self._file_path))
            is_doc = extension == ".doc"

        # 检查版本要求：.doc 文件需要 unstructured>=0.4.11
        if is_doc and unstructured_version < (0, 4, 11):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning .doc files is only supported in unstructured>=0.4.11. "
                "Please upgrade the unstructured package and try again."
            )

        # 根据文件类型选择不同的解析方法
        if is_doc:
            # .doc 文件：通过 API 解析（需要外部服务）
            # 原因：.doc 是微软的专有二进制格式，解析复杂度高，需要依赖重型库
            # 为了避免在主应用中引入过多依赖，使用外部 API 服务进行解析
            from unstructured.partition.api import partition_via_api
            elements = partition_via_api(filename=self._file_path, api_url=self._api_url, api_key=self._api_key)
        else:
            # .docx 文件：本地解析
            # 原因：.docx 是基于 XML 的开放格式，结构清晰，可以用轻量级库直接解析
            # 本地处理速度快，不依赖外部服务，适合频繁使用
            from unstructured.partition.docx import partition_docx
            elements = partition_docx(filename=self._file_path)

        # 按标题分块，优化文档结构
        from unstructured.chunking.title import chunk_by_title
        chunks = chunk_by_title(elements, max_characters=2000, combine_text_under_n_chars=2000)
        
        # 转换为 Document 对象
        documents = []
        for chunk in chunks:
            text = chunk.text.strip()
            documents.append(Document(page_content=text))
        return documents
