from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import Any, List, Optional
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


class CustomFlagEmbeddingReranker(FlagEmbeddingReranker):
    """Custom Flag Embedding Reranker to use paraphrase-specific queries."""

    def __init__(
        self,
        top_n: int = 2,
        model: str = "AITeamVN/Vietnamese_Reranker",
        use_fp16: bool = False,
    ) -> None:
        super().__init__(top_n=top_n, model=model, use_fp16=use_fp16)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Override the method to use the correct query for reranking."""

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        # Use the best query corresponding to each node
        query_and_nodes = [
            (
                query_bundle.query_str,  # Get the best query for the node
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]

        # Perform reranking using the correct query
        scores = self._model.compute_score(query_and_nodes)

        # If scores are returned as a single float, convert to list
        if isinstance(scores, float):
            scores = [scores]

        # Ensure the number of scores matches the number of nodes
        assert len(scores) == len(nodes)

        # Assign the scores to the nodes
        for node, score in zip(nodes, scores):
            node.score = score

        # Sort the nodes by score and return the top N
        new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[:self.top_n]

        # Return the processed nodes
        return new_nodes
