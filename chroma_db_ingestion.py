import os
import uuid
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

def setup_chroma_database(db_path: str = "./.chroma_db_4"):

    client = chromadb.PersistentClient(path=os.environ["CHROMA_DB_PATH"])
    print(f"ChromaDB client initialized at: {os.environ["CHROMA_DB_PATH"]}")

    documents_collection = client.get_or_create_collection(
        name="documents",
        # embedding_function=OpenAIEmbeddingFunction(
        #     api_key=os.environ["OPENAI_API_KEY"],
        #     model_name=os.environ["model"]
        # )
    )
    summaries_collection = client.get_or_create_collection(
        name="summaries",
        # embedding_function=OpenAIEmbeddingFunction(
        #     api_key=os.environ["OPENAI_API_KEY"],
        #     model_name=os.environ["model"]
        # )
    )
    
    urls = [
        "https://theportal.group/optics/",
        "https://theportal.group/35-superposition/",
        "https://theportal.group/30-the-awakening/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    raw_documents=[doc[0] for doc in docs]
    for doc in raw_documents:
        non_empty_lines = []
        for line in doc.page_content.splitlines():
            if line.strip():
                non_empty_lines.append(line)

        doc.page_content = '\n'.join(non_empty_lines)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(raw_documents)
    print(len(doc_splits))

    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | ChatOpenAI(model=os.environ["model"],max_retries=0)
        | StrOutputParser()
    )

    document_summaries = chain.batch(doc_splits, {"max_concurrency": 5})

    doc_ids = [str(uuid.uuid4()) for _ in range(len(doc_splits))]

    s=summaries_collection.add(
        documents=document_summaries,
        ids=doc_ids,
        metadatas=[{"doc_id": doc_id} for doc_id in doc_ids]
    )

    d=documents_collection.add(
        documents=[doc.page_content for doc in doc_splits],
        ids=doc_ids,
        metadatas=[{"doc_id": doc_id} for doc_id in doc_ids]
    )
    print("Database population complete!")
    return client

if __name__ == "__main__":
    db_client = setup_chroma_database()