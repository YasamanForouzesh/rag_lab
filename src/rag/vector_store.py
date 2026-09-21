from vector_db import Chroma




chroma = Chroma()


def create_embedding(chunks):
    chroma.add(chunks)


