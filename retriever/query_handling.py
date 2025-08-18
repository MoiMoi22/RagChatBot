from router.parser import CustomOutputParser
from router.schemas import AnswerQuery
from router.prompt import query_gen_prompt, FORMAT_OUTPUT_ANSWER_QUERIES
from typing import Dict, List, Tuple
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
import numpy as np


def generate_queries(llm, query_str: str, num_queries: int = 4):
    output_parser = CustomOutputParser(AnswerQuery)
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1,
        query=query_str
    )

    fmt_json_prompt = output_parser.format(fmt_prompt, FORMAT_OUTPUT_ANSWER_QUERIES)

    raw_output = llm.complete(fmt_json_prompt)
    parsed = output_parser.parse(str(raw_output))
    return parsed


def run_queries(
    queries,
    retrievers,
    embed_model
) -> Dict[Tuple[str, int], List["NodeWithScore"]]:
    """
    Chạy tuần tự qua từng query và retriever.
    Trả về dict: {(query, retriever_idx): List[NodeWithScore]}
    """
    results: Dict[Tuple[str, int], List["NodeWithScore"]] = {}

    for query in queries:
        # Chuẩn bị QueryBundle (embed trước nếu cần)
        query_embedding = embed_model.get_query_embedding(query)
        query_bundle = QueryBundle(query_str=query, embedding=query_embedding)

        for idx, retriever in enumerate(retrievers):
            if isinstance(retriever, BM25Retriever):
                # BM25: gọi sync và lọc theo department_id
                result_bm25 = retriever.retrieve(query_bundle)
                # result = [
                #     r for r in result_bm25
                #     if r.node.metadata.get("department_id") == user_department_id
                # ]
                result = result_bm25
            else:
                # Vector retriever nội bộ: ưu tiên _retrieve nếu cần truyền thêm tham số
                if hasattr(retriever, "_retrieve"):
                    result = retriever._retrieve(
                        query_bundle=query_bundle
                    )
                else:
                    # Fallback: gọi retrieve chuẩn (nếu không hỗ trợ _retrieve)
                    result = retriever.retrieve(query_bundle)

            results[(query, idx)] = result

    return results

def fuse_results(
    results_dict: Dict[Tuple[str, int], List["NodeWithScore"]],
    similarity_top_k: int = 2,
    k: float = 60.0,
):
    """
    Fuse tất cả kết quả bằng Reciprocal Rank Fusion (RRF), không phân biệt theo query.
    Trả về top-K NodeWithScore với điểm = fused RRF score.

    Parameters
    ----------
    results_dict : Dict[(query_text, retriever_idx), List[NodeWithScore]]
        Mỗi value là 1 danh sách đã xếp hạng (score cao hơn = tốt hơn).
    similarity_top_k : int
        Số lượng kết quả cuối cùng.
    k : float
        Hằng số RRF.

    Ghi chú:
    - Gộp theo passage_id (node.node_id) để tránh đụng độ text trùng.
    - Representative node được chọn là node có original score cao nhất từng thấy
      cho cùng passage_id. Sau đó ta gán fused score vào field `score`.
    """

    fused_scores: Dict[str, float] = {}                # passage_id -> fused score
    rep_node: Dict[str, "NodeWithScore"] = {}          # passage_id -> representative NodeWithScore

    # Duyệt qua mọi bucket, nhưng không dùng query_text
    for nodes_with_scores in results_dict.values():
        # RRF dùng thứ hạng trong từng danh sách: sort theo score giảm dần
        sorted_nodes = sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)

        for rank, nws in enumerate(sorted_nodes):
            node = nws.node
            passage_id = getattr(node, "node_id", None)
            if passage_id is None:
                # Nếu không có node_id, fallback tạm bằng nội dung (ít gặp)
                passage_id = getattr(node, "get_content", lambda: str(node))()

            # Cộng dồn điểm RRF
            fused_scores[passage_id] = fused_scores.get(passage_id, 0.0) + 1.0 / (rank + k)

            # Chọn representative node có original score cao nhất
            cur_best = rep_node.get(passage_id)
            if (cur_best is None) or ((nws.score or 0.0) > (cur_best.score or 0.0)):
                rep_node[passage_id] = nws

    # Gán fused score vào representative node
    for pid, score in fused_scores.items():
        rep_node[pid].score = score

    # Sắp xếp theo fused score giảm dần và lấy top-K
    final_nodes: List["NodeWithScore"] = sorted(
        rep_node.values(), key=lambda x: x.score or 0.0, reverse=True
    )

    return final_nodes[:similarity_top_k]

def rerank_and_normalize(
    nodes: List[NodeWithScore],
    threshold: float = 0.8,
    m: int = 5
) -> List[NodeWithScore]:
    """
    Rerank nodes, normalize scores, and apply thresholding for final selection.
    
    Args:
        nodes: List of NodeWithScore objects to be reranked.
        threshold: The threshold to filter nodes after normalization (default: 0.8).
        top_k: The number of top nodes to return after fusion (default: 10).
        m: The number of top nodes to return after reranking (default: 5).
    
    Returns:
        List of top M NodeWithScore objects after reranking and normalization.
    """

    # Step 1: Apply z-score normalization on all nodes' scores
    scores = np.array([node.score for node in nodes])
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    normalized_scores = (scores - mean_score) / (std_score + 1e-8)

    # Apply sigmoid to normalized scores
    sigmoid_scores = 1 / (1 + np.exp(-normalized_scores))
    
    # Assign normalized sigmoid scores back to nodes
    for node, score in zip(nodes, sigmoid_scores):
        node.score = score

    # Step 2: Apply thresholding (remove nodes below the threshold)
    filtered_nodes = [node for node in nodes if node.score >= threshold]

    # Step 3: Sort nodes by score (descending order)
    reranked_nodes = sorted(filtered_nodes, key=lambda x: -x.score)

    # Step 4: Return top M nodes
    return reranked_nodes[:m]