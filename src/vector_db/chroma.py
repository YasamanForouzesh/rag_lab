from .vector_adapter import VectorStore
from chromadb import PersistentClient
from pathlib import Path
import os
from openai import OpenAI

collection_name = "docs"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
embedding_model= os.getenv("EMBEDING_MODEL")
class Chroma(VectorStore):
    def add(self, chunks: list):
        openai = OpenAI()
        chroma = PersistentClient(path=DB_NAME)
        if collection_name in [c.name for c in chroma.list_collections()]:
            chroma.delete_collection(collection_name)

        texts = [chunk.page_content for chunk in chunks]
        emb = openai.embeddings.create(model=embedding_model, input=texts).data
        vectors = [e.embedding for e in emb]

        collection = chroma.get_or_create_collection(collection_name)

        ids = [str(i) for i in range(len(chunks))]
        metas = [chunk.metadata for chunk in chunks]

        collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
        print(f"Vectorstore created with {collection.count()} documents")

    def search(self, embeding, top_k=5):
        return super().search(embeding, top_k)


        