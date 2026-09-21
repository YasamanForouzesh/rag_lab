
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list):
        pass
    @abstractmethod
    def search(self, embeding, top_k =5):
        pass