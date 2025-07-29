from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# llm api
from llama_index.llms.groq import Groq
#Load api key
from dotenv import load_dotenv
import os
import chromadb

# Model embedding
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Nếu index chưa tồn tại thì tạo
# def build_index():
#     documents = SimpleDirectoryReader("./data").load_data()
    
#     db = chromadb.PersistentClient(path="./index")
#     chroma_collection = db.get_or_create_collection("rag_collection")
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
#     return index

# Load index
def load_rag_index():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("Test")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, storage_context=storage_context)
    return index

# Load llm
def load_llm():
    load_dotenv()  # load biến môi trường từ file .env
    API_KEY = os.getenv("MY_API_KEY")
    llm = Groq(model="qwen/qwen3-32b", api_key=API_KEY)
    return llm
