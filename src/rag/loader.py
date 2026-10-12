
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_NAME = str(PROJECT_ROOT / "vector_db")
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge-base"

@dataclass
class Document:
    page_content: str
    metadata: dict[str, str]

def fetch_document():
    """ A homemade version of the LangchainLoader"""
    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    Document(
                        page_content=f.read(),
                        metadata={"source": file.as_posix(), "type": doc_type},
                    )
                )
    print(len(documents))
    return documents


