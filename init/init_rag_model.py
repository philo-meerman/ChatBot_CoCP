import logging
from config import Config
from utils.pdf_processor import extract_boek_2_text
from models.rag_model import RAGModel

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_rag_model():
    """
    Initialize the RAG model by extracting text from the PDF, processing it, 
    generating embeddings, and storing/loading the FAISS index.

    Returns:
    - rag_model (RAGModel): The initialized RAG model.
    """
    api_key = Config.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize the RAG model
    rag_model = RAGModel()

    # Extract text from the PDF
    pdf_text = extract_boek_2_text(Config.DATAPATH)

    # Process and chunk the text
    rag_model.chunk_text(pdf_text)

    # Generate embeddings and store them
    if Config.EMBED_REGEN == 1:
        rag_model.generate_embeddings()
        rag_model.store_embeddings()

    # Load embeddings (optional, if not already in memory)
    rag_model.load_embeddings()

    logger.info("RAG model initialized")

    return rag_model
