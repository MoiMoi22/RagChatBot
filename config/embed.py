from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def load_embed():
    return HuggingFaceEmbedding(model_name="AITeamVN/Vietnamese_Embedding")