import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

def get_vector_store(collection_name:str):
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=chroma_collection)