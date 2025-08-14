import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore

from core.indexing_runner import DocumentIsPausedError, IndexingRunner
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from models.dataset import Dataset, Document, DocumentSegment


@shared_task(queue="dataset")
def document_indexing_update_task(dataset_id: str, document_id: str):
    """
    文档配置更新后的异步重新索引任务
    当文档的处理规则、数据源等配置发生变更时，需要清理旧索引并重新构建
    
    执行流程：
    1. 清理文档的所有分段和向量索引
    2. 重新执行完整的文档处理流程（解析、清洗、分块、向量化、索引）
    
    :param dataset_id: 数据集ID
    :param document_id: 文档ID
    
    Usage: document_indexing_update_task.delay(dataset_id, document_id)
    """
    logging.info(click.style(f"Start update document: {document_id}", fg="green"))
    start_at = time.perf_counter()

    document = db.session.query(Document).where(Document.id == document_id, Document.dataset_id == dataset_id).first()

    if not document:
        logging.info(click.style(f"Document not found: {document_id}", fg="red"))
        db.session.close()
        return

    # 开始重新处理，更新文档状态为解析中
    document.indexing_status = "parsing"
    document.processing_started_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    db.session.commit()

    # 第一阶段：清理旧的文档分段和索引数据
    try:
        dataset = db.session.query(Dataset).where(Dataset.id == dataset_id).first()
        if not dataset:
            raise Exception("Dataset not found")

        index_type = document.doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()

        # 获取文档的所有分段
        segments = db.session.query(DocumentSegment).where(DocumentSegment.document_id == document_id).all()
        if segments:
            index_node_ids = [segment.index_node_id for segment in segments]

            # 从向量数据库中删除索引数据（包括关键词和子分块）
            index_processor.clean(dataset, index_node_ids, with_keywords=True, delete_child_chunks=True)

            # 从数据库中删除分段记录
            for segment in segments:
                db.session.delete(segment)
            db.session.commit()
        end_at = time.perf_counter()
        logging.info(
            click.style(
                "Cleaned document when document update data source or process rule: {} latency: {}".format(
                    document_id, end_at - start_at
                ),
                fg="green",
            )
        )
    except Exception:
        logging.exception("Cleaned document when document update data source or process rule failed")

    # 第二阶段：重新执行完整的文档索引流程
    try:
        # 使用 IndexingRunner 重新处理文档
        # 会按照新的配置重新执行：解析 -> 清洗 -> 分块 -> 向量化 -> 索引
        indexing_runner = IndexingRunner()
        indexing_runner.run([document])
        end_at = time.perf_counter()
        logging.info(click.style(f"update document: {document.id} latency: {end_at - start_at}", fg="green"))
    except DocumentIsPausedError as ex:
        logging.info(click.style(str(ex), fg="yellow"))
    except Exception:
        logging.exception("document_indexing_update_task failed, document_id: %s", document_id)
    finally:
        db.session.close()
