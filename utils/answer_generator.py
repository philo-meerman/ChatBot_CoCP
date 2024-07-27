import logging
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from utils.summarizer import summarize_history
from utils.api import set_openai_api_key
from config import Config

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_answer(query, rag_model, conversation_history=None):
    """
    Generate an answer to the query using the RAG model and OpenAI API.

    Parameters:
    - query (str): The user's query.
    - rag_model: The RAG model instance.
    - conversation_history (list): The conversation history, default is None.

    Returns:
    - answer (str): The generated answer from the LLM.
    - serializable_history (list): The updated conversation history in a serializable format.
    """
    # Set the OpenAI API key
    set_openai_api_key()

    # Log the received query
    logger.info("Received query:\n%s", query)

    # Initialize conversation history if it's not provided
    if conversation_history is None:
        conversation_history = []

    # Ensure conversation history is in the correct format
    conversation_history = [ChatMessage(**msg) if isinstance(msg, dict) else msg for msg in conversation_history]

    # Add the current query to the conversation history
    conversation_history.append(ChatMessage(role="user", content=query))

    # Summarize the conversation history
    summary = summarize_history(history=conversation_history)
    logger.info("Summarized conversation history for RAG model query:\n%s", summary)

    # Retrieve relevant chunks from the RAG model
    relevant_chunks = rag_model.get_relevant_chunks(summary, k=5)
    context = "\n\n".join(relevant_chunks)
    logger.info("Retrieved %d relevant chunks", len(relevant_chunks))
    for i, chunk in enumerate(relevant_chunks):
        logger.info("\nChunk %d:\n%s", i + 1, chunk[:200])  # Log the first 200 characters of each chunk

    # Prepare the messages for the LLM
    messages = [
        ChatMessage(role="system", content=Config.SYSTEM_ROLE),
        ChatMessage(role="system", content=f"Relevante context:\n{context}")
    ] + conversation_history
    logger.info("Prepared messages for LLM:")
    for msg in messages:
        logger.info("%s: %s", msg.role, msg.content[:200])  # Log the first 200 characters of each message

    # Initialize the OpenAI chat model
    llm = OpenAI(temperature=0, model=Config.CHAT_MODEL, max_tokens=Config.MAX_TOKENS)

    # Get the response from the chat model
    responses = llm.chat(messages)
    answer = responses.message.content
    logger.info("Generated answer:\n%s", answer)

    # Add the assistant's response to the conversation history
    conversation_history.append(ChatMessage(role="assistant", content=answer))

    # Convert the conversation history to a serializable format
    serializable_history = [{"role": msg.role, "content": msg.content} for msg in conversation_history]

    return answer, serializable_history
