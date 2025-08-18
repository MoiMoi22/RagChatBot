from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

def build_flag_reranker(model_id: str = "AITeamVN/Vietnamese_Reranker", top_n: int = 5):
    reranker = FlagEmbeddingReranker(model=model_id, top_n=top_n)
    return reranker
