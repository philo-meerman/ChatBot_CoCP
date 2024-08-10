from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
import re
from config import Config
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        Parse the penal code text into a hierarchical structure of chapters, titles, afdelingen, and articles.

        Parameters:
        - text (str): The penal code text.

        Returns:
        - chapters (list): List of parsed chapters with titles, afdelingen, and articles.
        - self.article_mapping (dict): A dictionary mapping article numbers to their content and metadata.
        """
        # Define regex patterns for hoofdstuk, titles, afdelingen, and articles
        hoofdstuk_pattern = re.compile(r"^HOOFDSTUK\s+\d+.*$", re.MULTILINE)
        title_pattern = re.compile(r"^TITEL\s+\d+.*$", re.MULTILINE)
        afdeling_pattern = re.compile(r"^AFDELING\s+\d+.*$", re.MULTILINE)
        article_pattern = re.compile(r"^Artikel\s+\d+\.\d+\.\d+", re.MULTILINE)

        chapters = []
        self.article_mapping = {}
        current_hoofdstuk = None
        current_title = None
        current_afdeling = None
        current_article = None

        def get_article_number(article_line):
            """Extract the article number from an article line."""
            match = re.search(r"\d+\.\d+\.\d+", article_line)
            return match.group() if match else None

        last_article_number = None

        lines = text.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Identify Hoofdstukken
            if hoofdstuk_pattern.match(line):
                current_hoofdstuk = {"hoofdstuk": line, "titles": [], "content": ""}
                chapters.append(current_hoofdstuk)
                current_title = None
                current_afdeling = None
                current_article = None
                last_article_number = None

            # Identify Titles within Hoofdstukken
            elif title_pattern.match(line):
                current_title = {
                    "title": line,
                    "afdelingen": [],
                    "articles": [],
                    "content": "",
                }
                if current_hoofdstuk is not None:
                    current_hoofdstuk["titles"].append(current_title)
                else:
                    # If no current hoofdstuk, create an empty one to contain the title
                    current_hoofdstuk = {
                        "hoofdstuk": "",
                        "titles": [current_title],
                        "content": "",
                    }
                    chapters.append(current_hoofdstuk)
                current_afdeling = None
                current_article = None
                last_article_number = None

            # Identify Afdelingen within Titles
            elif afdeling_pattern.match(line):
                if current_title is None:
                    # Ensure current_title is initialized to avoid NoneType error
                    current_title = {
                        "title": "",
                        "afdelingen": [],
                        "articles": [],
                        "content": "",
                    }
                    current_hoofdstuk["titles"].append(current_title)
                current_afdeling = {"afdeling": line, "articles": [], "content": ""}
                current_title["afdelingen"].append(current_afdeling)
                current_article = None
                last_article_number = None

            # Identify Articles within Afdelingen or directly under Titles
            elif article_pattern.match(line):
                article_number = get_article_number(line)

                # Check if the current article number logically follows the last article number
                if last_article_number:
                    last_parts = list(map(int, last_article_number.split(".")))
                    current_parts = list(map(int, article_number.split(".")))

                    # Only consider it a new article if the number increments logically
                    is_new_article = (
                        current_parts[0] > last_parts[0]
                        or (
                            current_parts[0] == last_parts[0]
                            and current_parts[1] > last_parts[1]
                        )
                        or (
                            current_parts[0] == last_parts[0]
                            and current_parts[1] == last_parts[1]
                            and current_parts[2] > last_parts[2]
                        )
                    )
                else:
                    is_new_article = True

                if is_new_article:
                    current_article = {"article": article_number, "content": line}

                    # Add article to article_mapping with metadata
                    self.article_mapping[article_number] = {
                        "content": line,
                        "chapter": (
                            current_hoofdstuk["hoofdstuk"] if current_hoofdstuk else ""
                        ),
                        "title": current_title["title"] if current_title else "",
                        "afdeling": (
                            current_afdeling["afdeling"] if current_afdeling else ""
                        ),
                    }

                    if current_afdeling:
                        current_afdeling["articles"].append(current_article)
                    else:
                        if current_title is None:
                            # Ensure current_title is initialized
                            current_title = {
                                "title": "",
                                "afdelingen": [],
                                "articles": [],
                                "content": "",
                            }
                            current_hoofdstuk["titles"].append(current_title)
                        current_title["articles"].append(current_article)
                    last_article_number = article_number
                else:
                    if current_article:
                        current_article["content"] += " " + line
                        self.article_mapping[last_article_number]["content"] += " " + line

            else:
                # Append content to the current context (chapter, title, afdeling, or article)
                if current_article:
                    current_article["content"] += " " + line
                    self.article_mapping[last_article_number]["content"] += " " + line
                elif current_afdeling:
                    current_afdeling["content"] += " " + line
                elif current_title:
                    current_title["content"] += " " + line
                elif current_hoofdstuk:
                    current_hoofdstuk["content"] += " " + line

            i += 1

        return chapters

    def chunk_penal_code(self, parsed_code, chunk_size, overlap):
        """
        Chunk the parsed penal code into smaller text chunks.

        Parameters:
        - parsed_code (list): Parsed chapters containing titles, afdelingen, and articles.
        - chunk_size (int): Size of the text chunks.
        - overlap (int): Overlap size between chunks.

        Returns:
        - chunks (list): List of text chunks.
        """
        chunks = []

        for chapter in parsed_code:
            chapter_text = chapter["hoofdstuk"]

            for title in chapter["titles"]:
                title_text = title["title"]

                for afdeling in title["afdelingen"]:
                    afdeling_text = afdeling["afdeling"]
                    articles = afdeling["articles"]

                    # Initialize current_chunk and current_length
                    current_chunk = f"{chapter_text}\n{title_text}\n{afdeling_text}\n"
                    current_length = len(current_chunk.split())

                    for article in articles:
                        article_text = f"{article['article']}\n{article['content']}"
                        article_length = len(article_text.split())

                        if current_length + article_length <= chunk_size:
                            current_chunk += f"\n{article_text}"
                            current_length += article_length
                        else:
                            chunks.append(current_chunk)
                            current_chunk = f"{chapter_text}\n{title_text}\n{afdeling_text}\n{article_text}"
                            current_length = len(current_chunk.split())

                            # Handle overlap
                            if len(current_chunk.split()) > chunk_size - overlap:
                                chunks.append(current_chunk)
                                current_chunk = " ".join(current_chunk.split()[-overlap:])
                                current_length = len(current_chunk.split())

                    if current_chunk:
                        chunks.append(current_chunk)

                # Handle articles directly under titles (without afdelingen)
                if not title["afdelingen"]:  # Check if there are no afdelingen
                    current_chunk = f"{chapter_text}\n{title_text}\n"  # Initialize chunk with chapter and title
                    current_length = len(current_chunk.split())

                    for article in title["articles"]:
                        article_text = f"{article['article']}\n{article['content']}"
                        article_length = len(article_text.split())

                        if current_length + article_length <= chunk_size:
                            current_chunk += f"\n{article_text}"
                            current_length += article_length
                        else:
                            chunks.append(current_chunk)
                            current_chunk = f"{chapter_text}\n{title_text}\n{article_text}"
                            current_length = len(current_chunk.split())

                            # Handle overlap
                            if len(current_chunk.split()) > chunk_size - overlap:
                                chunks.append(current_chunk)
                                current_chunk = " ".join(current_chunk.split()[-overlap:])
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

        # Filter results by minimum similarity threshold (inner product in this case)
        filtered_indices = [
            indices[0][i]
            for i in range(len(distances[0]))
            if distances[0][i] >= min_similarity
        ]

        # Print the similarity scores of the returned objects
        print("Similarity Scores of Returned Chunks:")
        for i in range(len(distances[0])):
            if distances[0][i] >= min_similarity:
                print(
                    f"Chunk Index: {indices[0][i]}, Similarity Score: {distances[0][i]:.4f}"
                )

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
        logging.info(
            f"Attempting to retrieve content for article number: {article_number}"
        )

        article = self.article_mapping.get(article_number)

        if article:
            logging.info(
                f"Content successfully retrieved for article number: {article_number}"
            )
            return article["content"]
        else:
            logging.warning(
                f"Article number {article_number} not found in article mapping."
            )
            return "Sorry, het gevraagde artikel kon niet worden gevonden."
