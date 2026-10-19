
from pathlib import Path

from rag.loader import fetch_document
from rag.chunker import create_chuncks_by_Langchain
from rag.vector_store import create_embedding

def main():
    documents = fetch_document()
        # 1. Print the total number of documents
    print("Total documents:", len(documents))

    # 2. Print the first document's content and metadata
    if documents:
        print("\n--- DOCUMENT CONTENT ---")
        print(documents[0].page_content[:500])

        print("\n--- DOCUMENT METADATA ---")
        print(documents[0].metadata)

    chuncks = create_chuncks_by_Langchain(documents=documents)
    print("Total chunks:", len(chuncks))

    # 2. Print the first chunk's content and metadata
    if chuncks:
        for i, chunk in enumerate(chuncks[:11]):  # Print first 11 chunks (indices 0 to 10)
            print(f"\n--- CHUNK {i} CONTENT ---")
            print(chunk.page_content[:500])

            print(f"\n--- CHUNK {i} METADATA ---")
            print(chunk.metadata)
    # create_embedding(chunks=chuncks)

if __name__ == "__main__":
    main()