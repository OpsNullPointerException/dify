import base64
import logging
from typing import Any, Optional, cast

import numpy as np
from sqlalchemy.exc import IntegrityError

from configs import dify_config
from core.entities.embedding_type import EmbeddingInputType
from core.model_manager import ModelInstance
from core.model_runtime.entities.model_entities import ModelPropertyKey
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.rag.embedding.embedding_base import Embeddings
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from libs import helper
from models.dataset import Embedding

logger = logging.getLogger(__name__)


class CacheEmbedding(Embeddings):
    def __init__(self, model_instance: ModelInstance, user: Optional[str] = None) -> None:
        self._model_instance = model_instance
        self._user = user

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量：基于数据库持久化存储，避免重复计算相同文本的向量"""
        # 初始化结果数组，预设为None便于后续填充
        text_embeddings: list[Any] = [None for _ in range(len(texts))]
        embedding_queue_indices = []
        
        # 步骤1：检查数据库存储，找出需要新计算的文本
        for i, text in enumerate(texts):
            # 基于文本内容生成SHA256哈希值作为唯一标识
            hash = helper.generate_text_hash(text)  # SHA256(text + "None")
            # 查询数据库中是否已有该文本的向量记录
            embedding = (
                db.session.query(Embedding)
                .filter_by(
                    model_name=self._model_instance.model, hash=hash, provider_name=self._model_instance.provider
                )
                .first()
            )
            if embedding:
                # 数据库命中：直接使用已存储的向量
                text_embeddings[i] = embedding.get_embedding()
            else:
                # 数据库未命中：标记为需要计算
                embedding_queue_indices.append(i)
                
        # 步骤2：批量计算未缓存的文本向量
        if embedding_queue_indices:
            embedding_queue_texts = [texts[i] for i in embedding_queue_indices]
            embedding_queue_embeddings = []
            try:
                # 获取模型支持的最大批处理大小
                model_type_instance = cast(TextEmbeddingModel, self._model_instance.model_type_instance)
                model_schema = model_type_instance.get_model_schema(
                    self._model_instance.model, self._model_instance.credentials
                )
                max_chunks = (
                    model_schema.model_properties[ModelPropertyKey.MAX_CHUNKS]
                    if model_schema and ModelPropertyKey.MAX_CHUNKS in model_schema.model_properties
                    else 1
                )
                
                # 按模型限制分批调用嵌入API
                for i in range(0, len(embedding_queue_texts), max_chunks):
                    batch_texts = embedding_queue_texts[i : i + max_chunks]

                    # 调用嵌入模型生成向量
                    embedding_result = self._model_instance.invoke_text_embedding(
                        texts=batch_texts, user=self._user, input_type=EmbeddingInputType.DOCUMENT
                    )

                    # 处理每个向量：归一化并验证有效性
                    for vector in embedding_result.embeddings:
                        try:
                            # 向量归一化：转换为单位向量，提高相似度计算精度
                            normalized_embedding = (vector / np.linalg.norm(vector)).tolist()  # type: ignore
                            # 检查向量是否包含NaN值（无效计算结果）
                            if np.isnan(normalized_embedding).any():
                                logger.warning("Normalized embedding is nan: %s", normalized_embedding)
                                continue
                            embedding_queue_embeddings.append(normalized_embedding)
                        except IntegrityError:
                            db.session.rollback()
                        except Exception:
                            logging.exception("Failed transform embedding")
                            
                # 步骤3：保存新计算的向量到数据库
                cache_embeddings = []
                try:
                    for i, n_embedding in zip(embedding_queue_indices, embedding_queue_embeddings):
                        text_embeddings[i] = n_embedding
                        hash = helper.generate_text_hash(texts[i])
                        if hash not in cache_embeddings:
                            # 创建数据库记录：持久化存储向量数据
                            embedding_cache = Embedding(
                                model_name=self._model_instance.model,
                                hash=hash,
                                provider_name=self._model_instance.provider,
                            )
                            embedding_cache.set_embedding(n_embedding)
                            db.session.add(embedding_cache)
                            cache_embeddings.append(hash)
                    db.session.commit()
                except IntegrityError:
                    db.session.rollback()
            except Exception as ex:
                db.session.rollback()
                logger.exception("Failed to embed documents: %s")
                raise ex

        return text_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        # use doc embedding cache or store if not exists
        hash = helper.generate_text_hash(text)
        embedding_cache_key = f"{self._model_instance.provider}_{self._model_instance.model}_{hash}"
        embedding = redis_client.get(embedding_cache_key)
        if embedding:
            redis_client.expire(embedding_cache_key, 600)
            decoded_embedding = np.frombuffer(base64.b64decode(embedding), dtype="float")
            return [float(x) for x in decoded_embedding]
        try:
            embedding_result = self._model_instance.invoke_text_embedding(
                texts=[text], user=self._user, input_type=EmbeddingInputType.QUERY
            )

            embedding_results = embedding_result.embeddings[0]
            # FIXME: type ignore for numpy here
            embedding_results = (embedding_results / np.linalg.norm(embedding_results)).tolist()  # type: ignore
            if np.isnan(embedding_results).any():
                raise ValueError("Normalized embedding is nan please try again")
        except Exception as ex:
            if dify_config.DEBUG:
                logging.exception("Failed to embed query text '%s...(%s chars)'", text[:10], len(text))
            raise ex

        try:
            # encode embedding to base64
            embedding_vector = np.array(embedding_results)
            vector_bytes = embedding_vector.tobytes()
            # Transform to Base64
            encoded_vector = base64.b64encode(vector_bytes)
            # Transform to string
            encoded_str = encoded_vector.decode("utf-8")
            redis_client.setex(embedding_cache_key, 600, encoded_str)
        except Exception as ex:
            if dify_config.DEBUG:
                logging.exception(
                    "Failed to add embedding to redis for the text '%s...(%s chars)'", text[:10], len(text)
                )
            raise ex

        return embedding_results  # type: ignore
