# models/rag_model.py
import openai
from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import re


load_dotenv()

api_key = os.getenv("OPENAI_AI_KEY")

class RAGModel:
    def __init__(self, chunk_size=1024, chunk_overlap = 50, index_path="embeddings.index"):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path
        self.chunks = []
        self.embeddings = []
        openai.api_key = self.api_key

    def parse_penal_code(self, text):
        # Define regex patterns for titles, afdelingen, and articles
        title_pattern = re.compile(r'^TITEL\s+\d+.*$', re.MULTILINE)
        afdeling_pattern = re.compile(r'^AFDELING\s+\d+.*$', re.MULTILINE)
        article_pattern = re.compile(r'^Artikel\s+\d+\.\d+\.\d+', re.MULTILINE)
        
        titles = []
        current_title = None
        current_afdeling = None

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if title_pattern.match(line):
                current_title = {'title': line, 'afdelingen': []}
                titles.append(current_title)
                current_afdeling = None
            elif afdeling_pattern.match(line):
                current_afdeling = {'afdeling': line, 'articles': []}
                current_title['afdelingen'].append(current_afdeling)
            elif article_pattern.match(line):
                if current_afdeling is None:
                    # Handle case where article appears without afdeling
                    current_afdeling = {'afdeling': '', 'articles': []}
                    current_title['afdelingen'].append(current_afdeling)
                current_article = {'article': line, 'content': ''}
                current_afdeling['articles'].append(current_article)
                i += 1
                while i < len(lines) and not article_pattern.match(lines[i]) and not afdeling_pattern.match(lines[i]) and not title_pattern.match(lines[i]):
                    current_article['content'] += lines[i].strip() + ' '
                    i += 1
                continue
            i += 1

        return titles
    
    def chunk_penal_code(self, parsed_code, chunk_size=1000, overlap=50):
        chunks = []
        for title in parsed_code:
            title_text = title['title']
            for afdeling in title['afdelingen']:
                afdeling_text = afdeling['afdeling']
                articles = afdeling['articles']
                
                current_chunk = f"{title_text}\n{afdeling_text}\n"
                current_length = len(current_chunk.split())

                for article in articles:
                    article_text = f"{article['article']}\n{article['content']}"
                    article_length = len(article_text.split())
                    if current_length + article_length <= chunk_size:
                        current_chunk += f"\n{article_text}"
                        current_length += article_length
                    else:
                        chunks.append(current_chunk)
                        current_chunk = f"{title_text}\n{afdeling_text}\n{article_text}"
                        current_length = len(current_chunk.split())
                        
                        # Handle overlap
                        if len(current_chunk.split()) > chunk_size - overlap:
                            chunks.append(current_chunk)
                            current_chunk = ' '.join(current_chunk.split()[-overlap:])
                            current_length = len(current_chunk.split())
                
                if current_chunk:
                    chunks.append(current_chunk)
        
        return chunks

    def chunk_text(self, text):

        parsed_penal_code = self.parse_penal_code(text)
        chunks = self.chunk_penal_code(parsed_penal_code)

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

