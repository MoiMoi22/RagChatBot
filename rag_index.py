from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
# llm api
from llama_index.llms.groq import Groq
#Load api key
from dotenv import load_dotenv
import os
import chromadb


# Load index
def load_rag_index(collection_name:str, embed_model):
    vector_store = get_vector_store(collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, storage_context=storage_context)
    return index

def get_vector_store(collection_name:str):
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=chroma_collection)

def build_index(path:str, collection_name:str, embed_model):

    # load documents & add doc_id metadata
    documents = SimpleDirectoryReader(path).load_data()
    for doc in documents:
        file_name = doc.metadata.get("file_name")
        doc_id = int(file_name.split(".")[0])
        doc.metadata["doc_id"] = doc_id
    
    # Text splitter
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    # chunking & indexing
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(documents,
                            storage_context=storage_context,
                            embed_model=embed_model,
                            text_splitter = text_splitter
                            )
    return index


# Load llm
def load_llm():
    load_dotenv()  # load biến môi trường từ file .env
    API_KEY = os.getenv("MY_API_KEY")
    llm = Groq(model="qwen/qwen3-32b", api_key=API_KEY)
    return llm

# Load model embedding
def load_embed():
    return HuggingFaceEmbedding(model_name="AITeamVN/Vietnamese_Embedding")
