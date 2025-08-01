from llama_index.core.schema import NodeWithScore

def get_doc_ids(source_nodes:list[NodeWithScore]):
    doc_ids = list(set([
    node.metadata.get("doc_id")
    for node in source_nodes
    if node.metadata.get("doc_id") is not None
    ]))
    return doc_ids