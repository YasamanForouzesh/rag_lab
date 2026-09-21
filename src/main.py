
from pathlib import Path

from rag.loader import fetch_document
from rag.chunker import create_chunks


def main():
    documents = fetch_document()
    print(create_chunks(documents))

if __name__ == "__main__":
    print(Path(__file__).resolve().parents[1], "---------------")
    main()