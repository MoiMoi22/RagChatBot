from llama_index.core.llms import ChatMessage
from llama_index.core.base.response.schema import Response
from retriever.custom_retriever import ChromaDBRetriever, FusionRetriever
from retriever.custom_query_engine import DepartmentAwareQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from utils.utils import vn_tokenizer_no_stopword


def handle_chitchat(query: str, llm) -> str:
    """Xử lý câu hỏi chitchat bằng LLM."""
    messages = [
        ChatMessage(
            role="system",
            content=(
                "Bạn là một trợ lý ảo thân thiện, thông minh và hài hước. "
                "Hãy trả lời người dùng một cách tự nhiên, gần gũi và dễ hiểu, "
                "giống như đang trò chuyện với một người bạn. "
                "Tránh dùng ngôn ngữ kỹ thuật hay quá máy móc. "
                "Câu trả lời nên ngắn gọn, sinh động và có cảm xúc nếu phù hợp."
            )
        ),
        ChatMessage(
            role="user",
            content=query
        )
    ]

    response = llm.chat(messages)
    return Response(response= "CHITCHAT :" + response.message.content.strip(), metadata={"doc_ids": None})

def handle_departments_req(vector_store, embed_model, user_department_id, llm, reranker, query_str: str):

    vector_retriever = ChromaDBRetriever(vector_store=vector_store, embed_model=embed_model, similarity_top_k= 10)
    bm25_retriever = load_bm25_retriever(vector_store)
    retriever = FusionRetriever(
    llm, [vector_retriever, bm25_retriever],embed_model, similarity_top_k=20
)
    custom_query_engine = DepartmentAwareQueryEngine(
        retriever=retriever,
        llm=llm,
        user_department_id= user_department_id,
        reranker= reranker
    )
    # query_str = "lịch học skill writing ở tuần 1 thế nào?"
    query_embedding = embed_model.get_query_embedding(query_str)
    query_bundle = QueryBundle(query_str=query_str, embedding=query_embedding)
    response = custom_query_engine.query(query_bundle)
    return response


def load_bm25_retriever(vector_store):
    nodes = vector_store._get(limit=1000000, where=()).nodes
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=10,
        stemmer=None,
        skip_stemming=True,
        tokenizer=vn_tokenizer_no_stopword    
    )
    return bm25_retriever
