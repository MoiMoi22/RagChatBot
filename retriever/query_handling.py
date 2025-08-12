from router.parser import CustomOutputParser
from router.schemas import AnswerQuery
from router.prompt import query_gen_prompt, FORMAT_OUTPUT_ANSWER_QUERIES
from typing import Dict, List, Tuple
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever


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
    embed_model,
    user_department_id
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
                result = [
                    r for r in result_bm25
                    if r.node.metadata.get("department_id") == user_department_id
                ]
            else:
                # Vector retriever nội bộ: ưu tiên _retrieve nếu cần truyền thêm tham số
                if hasattr(retriever, "_retrieve"):
                    result = retriever._retrieve(
                        query_bundle=query_bundle,
                        user_department_id=user_department_id
                    )
                else:
                    # Fallback: gọi retrieve chuẩn (nếu không hỗ trợ _retrieve)
                    result = retriever.retrieve(query_bundle)

            results[(query, idx)] = result

    return results

def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]