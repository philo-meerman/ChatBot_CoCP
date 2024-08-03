from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
import re
from config import Config

class RAGModel:
    """
    A class to represent a Retrieval-Augmented Generation (RAG) model.
    
    Attributes:
    - api_key (str): API key for accessing OpenAI services.
    - chunk_size (int): Size of the text chunks.
    - chunk_overlap (int): Overlap size between chunks.
    - index_path (str): Path to store the FAISS index.
    - chunks (list): List of text chunks.
    - embeddings (list): List of embeddings for the text chunks.
    """

    def __init__(self, 
                 chunk_size=Config.CHUNK_SIZE, 
                 chunk_overlap=Config.CHUNK_OVERLAP, 
                 index_path="embeddings.index"
                 ):
        """
        Initialize the RAG model with the given parameters.
        
        Parameters:
        - chunk_size (int): Size of the text chunks.
        - chunk_overlap (int): Overlap size between chunks.
        - index_path (str): Path to store the FAISS index.
        """
        self.api_key = Config.OPENAI_API_KEY
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path
        self.chunks = []
        self.embeddings = []
        self.article_mapping = {}

    def parse_penal_code(self, text):
        """
        Parse the penal code text into titles, afdelingen, and articles.
        
        Parameters:
        - text (str): The penal code text.
        
        Returns:
        - titles (list): List of parsed titles with afdelingen and articles.
        """
        # Define regex patterns for titles, afdelingen, and articles
        title_pattern = re.compile(r'^TITEL\s+\d+.*$', re.MULTILINE)
        afdeling_pattern = re.compile(r'^AFDELING\s+\d+.*$', re.MULTILINE)
        article_pattern = re.compile(r'^Artikel\s+\d+\.\d+\.\d+', re.MULTILINE)
        sub_element_pattern = re.compile(r'^[a-z]\.\s')  # Matches sub-elements like a., b., etc.

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
                article_number = line.split()[1]
                article_content_lines = []
                i += 1
                while (
                    i < len(lines) and 
                    not article_pattern.match(lines[i]) and 
                    not afdeling_pattern.match(lines[i]) and 
                    not title_pattern.match(lines[i])
                    ):
                    # Preserve new lines and indent sub-elements
                    stripped_line = lines[i].strip()
                    if sub_element_pattern.match(stripped_line):
                        article_content_lines.append('    ' + stripped_line)  # Indent sub-elements
                    else:
                        article_content_lines.append(stripped_line)
                    i += 1
                current_article['content'] = "\n".join(article_content_lines) 
                self.article_mapping[article_number] = current_article  
                continue
            i += 1

        return titles

    def chunk_penal_code(self, 
                         parsed_code, 
                         chunk_size, 
                         overlap
                         ):
        """
        Chunk the parsed penal code into smaller text chunks.
        
        Parameters:
        - parsed_code (list): Parsed titles, afdelingen, and articles.
        - chunk_size (int): Size of the text chunks.
        - overlap (int): Overlap size between chunks.
        
        Returns:
        - chunks (list): List of text chunks.
        """
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
        """
        Chunk the text into smaller chunks.
        
        Parameters:
        - text (str): The text to be chunked.
        
        Returns:
        - chunks (list): List of text chunks.
        """
        parsed_penal_code = self.parse_penal_code(text)
        chunks = self.chunk_penal_code(parsed_code=parsed_penal_code, 
                                       chunk_size=self.chunk_size, 
                                       overlap=self.chunk_overlap
                                       )
        self.chunks = chunks
        return chunks

    def generate_embeddings(self):
        """
        Generate embeddings for the text chunks.
        
        Returns:
        - embeddings (list): List of embeddings for the text chunks.
        """
        embeddings = []
        embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key, 
                                           model=Config.EMBED_MODEL
                                           )

        for chunk in self.chunks:
            response = embedding_model.embed_query(chunk)
            embeddings.append(response)

        self.embeddings = embeddings
        return embeddings

    def store_embeddings(self):
        """
        Store the generated embeddings in a FAISS index.
        """
        dimension = len(self.embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(self.embeddings))
        faiss.write_index(index, self.index_path)

    def load_embeddings(self):
        """
        Load the FAISS index with stored embeddings.
        
        Returns:
        - index: The loaded FAISS index.
        """
        self.index = faiss.read_index(self.index_path)
        return self.index

    def query_embeddings(self, 
                         query, 
                         k=Config.TOP_K, 
                         min_similarity=Config.MIN_SIMILARITY
                         ):
        """
        Query the FAISS index with a given query to find the top-k most similar chunks
        with a minimum similarity threshold.
        
        Parameters:
        - query (str): The query text.
        - k (int): Number of top similar chunks to retrieve.
        - min_similarity (float): Minimum similarity threshold (between 0 and 1).
        
        Returns:
        - indices (list): List of indices of the similar chunks meeting the similarity threshold.
        """
        # Embed the query
        embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key, model=Config.EMBED_MODEL)
        response = embedding_model.embed_query(query)
        query_embedding = np.array([response], dtype=np.float32)

        # Normalize the query embedding for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, k)
        print(distances)
        # Filter results by minimum similarity threshold (inner product in this case)
        filtered_indices = [
            indices[0][i]
            for i in range(len(distances[0]))
            if distances[0][i] >= min_similarity
        ]

        return filtered_indices

    def get_relevant_chunks(self, query, k=Config.TOP_K):
        """
        Get the relevant text chunks for a given query.
        
        Parameters:
        - query (str): The query text.
        - k (int): Number of top similar chunks to retrieve.
        
        Returns:
        - relevant_chunks (list): List of relevant text chunks.
        """
        indices = self.query_embeddings(query, k)
        relevant_chunks = [
            self.chunks[idx] for idx in indices if idx < len(self.chunks)
        ]
        return relevant_chunks

    def get_exact_article(self, article_number):
        """
        Retrieve the exact article content based on the article number.

        Parameters:
        - article_number (str): The article number to retrieve.

        Returns:
        - str: The content of the specified article or an error message if not found.
        """
        article = self.article_mapping.get(article_number)
        if article:
            return article['content']
        else:
            return "Sorry, het gevraagde artikel kon niet worden gevonden."
