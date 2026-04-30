"""Retriever implementations (embedding-based dense retrieval).

Re-exports:
    ContrieverRetriever — facebook/contriever (default, matches PoisonedRAG)
    BGERetriever       — BAAI/bge-base-en-v1.5
"""

from src.retrievers.contriever import ContrieverRetriever
from src.retrievers.bge import BGERetriever

__all__ = ["ContrieverRetriever", "BGERetriever"]
