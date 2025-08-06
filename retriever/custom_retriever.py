from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing import Optional


class ChromaDBRetriever(BaseRetriever):
    """Retriever over a chroma vector store."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle, user_department_id: str) -> List[NodeWithScore]:
        """Retrieve nodes with department filtering logic."""
        if query_bundle.embedding is None:
            query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        else:
            query_embedding = query_bundle.embedding

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode
        )
        query_result = self._vector_store.query(vector_store_query)

        # Danh sách node + điểm
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        # === Custom logic ===
        matched = [n for n in nodes_with_scores if n.node.metadata.get("department_id") == user_department_id]
        unmatched = [n for n in nodes_with_scores if n.node.metadata.get("department_id") != user_department_id]

        if len(nodes_with_scores) == 0:
            # Case 3: Không có node nào được retrieve
            self._case = "no_result"
            return []

        if len(matched) == 0 and len(unmatched) > 0:
            # Case 4: Có node nhưng không thuộc phòng ban
            self._case = "wrong_department"
            return []

        if len(matched) == len(nodes_with_scores):
            # Case 1: Toàn bộ đúng phòng ban
            self._case = "all_match"
            return matched

        # Case 2: Một phần đúng, một phần sai
        self._case = "partial_match"
        return matched
