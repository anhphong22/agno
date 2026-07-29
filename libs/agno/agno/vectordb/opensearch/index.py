from enum import Enum


class SpaceType(str, Enum):
    l2: str = "l2"
    cosinesimil: str = "cosinesimil"
    innerproduct: str = "innerproduct"


class Engine(str, Enum):
    # nmslib is deprecated and rejected for new index creation from OpenSearch 3.0.0 onwards.
    # It is kept for compatibility with clusters running OpenSearch 2.x.
    nmslib: str = "nmslib"
    faiss: str = "faiss"
    lucene: str = "lucene"
