from dotenv import load_dotenv
import os
load_dotenv()
pythonpath = os.getenv('PYTHONPATH')
print("pythonpath", pythonpath)

import sys
# sys.path.append(os.path.dirname('d:\\greg\\langgraph_2\\langgraph-course\\graph'))
# print("sys.path", sys.path)

from pprint import pprint
from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "medium is the message"
    docs = retriever.invoke(question)
    if len(docs) == 0:
        print("docs", docs, type(docs))
        print("no docs found")
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "What does the medium is the message mean?", "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "medium is the message mean"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"
    
    
def test_generation_chain() -> None:
    question = "medium is the message"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)