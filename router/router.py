from typing import List
from router.prompt import *
from router.handler import *
from router.parser import RouterOutputParser
from llama_index.core.base.response.schema import Response
from llama_index.vector_stores.chroma import ChromaVectorStore
from utils.utils import get_choice_str, extract_choices, extract_answer


def route_query(
    query_str: str, choices: List[str], output_parser: RouterOutputParser, llm
):
    choices_str = get_choice_str(choices)

    fmt_base_prompt = router_prompt0.format(
        num_choices=len(choices),
        max_outputs=len(choices),
        context_list=choices_str,
        query_str=query_str,
    )
    fmt_json_prompt = output_parser.format(fmt_base_prompt)

    raw_output = llm.complete(fmt_json_prompt)
    parsed = output_parser.parse(str(raw_output))

    return parsed

def route_by_choice(choices: List[int], query_str:str, user_department_id:int, vector_store, embed_model, llm):
    if not choices:
        return Response(response="Không có lựa chọn nào được xác định.",
                metadata={"doc_ids": None})

    if len(choices) == 1:
        choice = choices[0]
        if choice == 1:
            return handle_departments_req(vector_store, embed_model, user_department_id,llm, query_str)
        elif choice == 2:
            return handle_chitchat(query=query_str, llm = llm)
        else:
            return Response(response=f"Lựa chọn không được hỗ trợ: {choice}",
                            metadata={"doc_ids": None})
    elif len(choices) == 2:
            return handle_departments_req(vector_store, embed_model, user_department_id,llm, query_str)
    else:
        return Response(response=f"Trường hợp không xử lý: nhiều hơn 2 lựa chọn ({choices})",
                        metadata={"doc_ids": None})

def routing(query_str:str,  user_department_id:int, vector_store: ChromaVectorStore, embed_model, llm):
    output_parser = RouterOutputParser()
    llm_choices = route_query(query_str, choices, output_parser, llm)

    extracted_choices = extract_choices(llm_choices)
    raw_answer = route_by_choice(extracted_choices, query_str, user_department_id, vector_store, embed_model, llm)
    return extract_answer(raw_answer)