from router.parser import CustomOutputParser
from router.schemas import AnswerQuery
from router.prompt import QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL, FORMAT_OUTPUT_ANSWER_QUERIES
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from typing import List
from llama_index.core.schema import BaseNode, MetadataMode
from utils.utils import extract_queries
import uuid
from tqdm import tqdm
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from retriever.query_handling import generate_queries

def generate_eval_queries(nodes: List[BaseNode], llm, num_questions_per_chunk: int = 2):
    output_parser = CustomOutputParser(AnswerQuery)
    fmt_prompt_system = output_parser.format(QUESTION_GEN_SYS_TMPL, FORMAT_OUTPUT_ANSWER_QUERIES)
    question_gen_template = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=fmt_prompt_system),
        ChatMessage(role=MessageRole.USER, content=QUESTION_GEN_USER_TMPL),
    ]
    )
    """Generate questions."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(node_dict.items()):
        fmt_messages = question_gen_template.format_messages(
            num_questions_per_chunk=2,
            context_str=text,
        )
        chat_response = llm.chat(fmt_messages)
        parsed = output_parser.parse(str(chat_response))
        questions = extract_queries(parsed)
        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    return EmbeddingQAFinetuneDataset(
            queries=queries, corpus=node_dict, relevant_docs=relevant_docs
            )

def map_ori_generated(queries:dict, llm):
    generated_queries = {}
    for query_id,query in queries.items():
        raw_queries = generate_queries(llm, query, 4)
        cur_queries = extract_queries(raw_queries)
        generated_queries[query_id] = cur_queries
    return generated_queries