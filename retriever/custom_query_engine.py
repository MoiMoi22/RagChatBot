from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryBundle
from retriever.custom_retriever import ChromaDBRetriever
from retriever.query_handling import rerank_and_normalize

class DepartmentAwareQueryEngine(BaseQueryEngine):
    def __init__(self, retriever: ChromaDBRetriever, llm, user_department_id: int, reranker = None, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self.retriever = retriever
        self.llm = llm
        self.user_department_id = user_department_id
        self.reranker = reranker

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes = self.retriever._retrieve(query_bundle)

        rerank_nodes = self.reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
        # rerank_nodes = rerank_and_normalize(rerank_nodes,0.8,3)

        for n in rerank_nodes:
            print(n.score)

        matched = [n for n in rerank_nodes if n.node.metadata.get("department_id") == self.user_department_id]
        unmatched = [n for n in rerank_nodes if n.node.metadata.get("department_id") != self.user_department_id]

        print(len(matched))
        print(len(unmatched))

        if len(rerank_nodes) == 0:
            return Response(response="RAG: " + "Không tìm thấy thông tin nào liên quan trong toàn bộ tài liệu.",
                            metadata={"doc_ids": None})


        if len(matched) == 0 and len(unmatched) > 0:
            return Response(response="RAG: " + "Bạn không có quyền truy cập thông tin của phòng ban khác.",
                            metadata={"doc_ids": None})

        if len(matched) > 0:
                
            context = "\n".join([n.node.text for n in rerank_nodes])
            prompt = f"Trả lời câu hỏi không sử dụng thông tin đã được train từ trước chỉ dựa trên thông tin sau:\n\n{context}\n\nCâu hỏi: {query_bundle.query_str}"
            answer = self.llm.complete(prompt)

            doc_ids = list(set([
                n.node.metadata.get("doc_id")
                for n in matched
                if n.node.metadata.get("doc_id") is not None
            ])) or None

            return Response(response=answer, metadata={"doc_ids": doc_ids})

        else:
            return Response(response="RAG: " + "Không xác định được kết quả xử lý truy vấn.",
                            metadata={"doc_ids": None})

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        raise NotImplementedError("Async query not supported.")
    def _get_prompt_modules(self) -> dict:
        return {}

