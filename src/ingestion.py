
from pathlib import Path

from rag.loader import fetch_document
from rag.chunker import create_chuncks_by_Langchain
from rag.vector_store import create_embedding

def main():
    documents = fetch_document()
    chuncks = create_chuncks_by_Langchain(documents=documents)
    # create_embedding(chunks=chuncks)

if __name__ == "__main__":
    main()