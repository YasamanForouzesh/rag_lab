
from pathlib import Path

from rag.loader import fetch_document
from rag.chunker import create_chunks
from rag.vector_store import create_embedding

def main():
    documents = fetch_document()
    chuncks = create_chunks(documents=documents)
    create_embedding(chunks=chuncks)

if __name__ == "__main__":
    main()