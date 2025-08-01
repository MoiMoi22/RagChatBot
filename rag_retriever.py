from ChromaDBRetriever import ChromaDBRetriever

def get_retriever(vector_store, embed_model, query_mode="default", similarity_top_k=2):
    return ChromaDBRetriever(vector_store=vector_store, embed_model= embed_model, query_mode=query_mode, similarity_top_k= similarity_top_k)