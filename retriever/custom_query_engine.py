from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryBundle
from ChromaDBRetriever import ChromaDBRetriever

class DepartmentAwareQueryEngine(BaseQueryEngine):
    def __init__(self, retriever: ChromaDBRetriever, llm, user_department_id: int, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self.retriever = retriever
        self.llm = llm
        self.user_department_id = user_department_id

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes = self.retriever._retrieve(query_bundle, user_department_id=self.user_department_id)
        case = self.retriever._case

        if case == "no_result":
            return Response(response="RAG: " + "Không tìm thấy thông tin nào liên quan trong toàn bộ tài liệu.",
                            metadata={"doc_ids": None})

        elif case == "wrong_department":
            return Response(response="RAG: " + "Bạn không có quyền truy cập thông tin của phòng ban khác.",
                            metadata={"doc_ids": None})

        elif case in ["partial_match", "all_match"]:
            context = "\n".join([n.node.text for n in nodes])
            prompt = f"Trả lời câu hỏi dựa trên thông tin sau:\n\n{context}\n\nCâu hỏi: {query_bundle.query_str}"
            answer = self.llm.complete(prompt)

            doc_ids = list(set([
                n.node.metadata.get("doc_id")
                for n in nodes
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

