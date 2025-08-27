from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever
from chroma_db_retrieve import client, retrieve_document_from_summary

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]


    retrieved_info = retrieve_document_from_summary(client, question)
    if retrieved_info:
        # print("\n--- Retrieved Information ---")
        # print("Query:", question)
        # print("Relevant Summary:", retrieved_info['retrieved_summary'][:50])
        # print("Full Document:", retrieved_info['full_documents'][0].page_content[:100])            
        # print("Full Document111111:", type(retrieved_info['full_documents']))
        return {"documents": retrieved_info["full_documents"], "question": question}
    else:
        summary_docs = retriever.invoke(question)
        if len(summary_docs) > 0:
            return {"documents": summary_docs, "question": question}
        else:
            return {"documents": [], "question": question}