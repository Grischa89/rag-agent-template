import chromadb
from langchain.schema import Document
import os

def retrieve_document_from_summary(client, query: str):
    try:
        documents_collection = client.get_collection(
            name="documents",
            # embedding_function=OpenAIEmbeddingFunction(
            #     api_key=os.environ["OPENAI_API_KEY"],
            #     model_name=os.environ["model"]
            # )
        )
        summaries_collection = client.get_collection(
            name="summaries",
            # embedding_function=OpenAIEmbeddingFunction(
            #     api_key=os.environ["OPENAI_API_KEY"],
            #     model_name=os.environ["model"]
            # )
        )
    except ValueError:
        print("Error: Collections not found. Please run setup_chroma_database() first.")
        return None

    search_results = summaries_collection.query(
        query_texts=[query],
        n_results=3
    )
    if not search_results['metadatas'][0]:
        print("No results found for the query.")
        return None
    print("search_results:", search_results)
    retrieved_doc_ids = [ search_results['metadatas'][0][i]['doc_id'] for i in range(0,len(search_results['metadatas'][0])) ]
    print("retrieved_doc_ids: ", retrieved_doc_ids)
    retrieved_summaries = search_results['documents'][0]
    print("search_results['metadatas'][0][0]: ", search_results['metadatas'][0][0])
    full_document_data = documents_collection.get(ids=retrieved_doc_ids)
    print("full_document_data:", len(full_document_data), type(full_document_data))

    return {
        "full_documents": [ Document(page_content=doc) for doc in full_document_data['documents']],
        "retrieved_summary": retrieved_summaries
    }


db_path=os.environ["CHROMA_DB_PATH"]
client = chromadb.PersistentClient(path=db_path)
print(f"ChromaDB client initialized at: {db_path}")

# test_query = "What does the medium is the message mean?"
# print(f"\nSearching with query: '{test_query}'")

# retrieved_info = retrieve_document_from_summary(client, test_query)

# if retrieved_info:
#     print("\n--- Retrieved Information ---")
#     print("Query:", test_query)
#     print("Relevant Summary:", retrieved_info['retrieved_summary'].strip()[:50])
#     print("Full Document:", retrieved_info['full_document'][0].strip()[:50])
# else:
#     print("Retrieval failed.")