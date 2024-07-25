# models/rag_model.py
import openai
from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_AI_KEY")

class RAGModel:
    def __init__(self, chunk_size=1000, index_path="embeddings.index"):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.index_path = index_path
        self.chunks = []
        self.embeddings = []
        openai.api_key = self.api_key

    def chunk_text(self, text):
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentence]
                current_length = sentence_length

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        self.chunks = chunks
        return chunks

    def generate_embeddings(self):
        embeddings = []
        embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)

        for chunk in self.chunks:
            response = embedding_model.embed_query(chunk)
            embeddings.append(response)

        self.embeddings = embeddings
        return embeddings

    def store_embeddings(self):
        dimension = len(self.embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(self.embeddings))
        faiss.write_index(index, self.index_path)

    def load_embeddings(self):
        self.index = faiss.read_index(self.index_path)
        return self.index

    def query_embeddings(self, query, k=50):
        embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)
        response = embedding_model.embed_query(query)
        query_embedding = np.array([response], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)
        print(indices[0])
        return indices[0]

    def get_relevant_chunks(self, query, k=5):
        indices = self.query_embeddings(query, k)
        relevant_chunks = [
            self.chunks[idx] for idx in indices if idx < len(self.chunks)
        ]
        return relevant_chunks

