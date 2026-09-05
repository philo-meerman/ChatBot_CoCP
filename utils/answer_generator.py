"""
RAG Model Module

This module implements a Retrieval-Augmented Generation (RAG) model tailored for processing
penal code texts. It provides functionality to parse structured legal texts, chunk them into
manageable pieces, generate embeddings for these chunks using OpenAI's embeddings, store and
load these embeddings with FAISS for efficient similarity search, and retrieve relevant text
chunks or specific articles based on a query.

The main class, RAGModel, offers the following capabilities:
    - Parsing penal code text into structured components such as titles, afdelingen, and articles.
    - Chunking the parsed text into smaller segments with configurable chunk sizes and overlaps.
    - Generating embeddings for text chunks via the OpenAIEmbeddings model.
    - Storing generated embeddings in a FAISS index and reloading them as needed.
    - Querying the FAISS index to obtain indices of the most similar text chunks for a given query.
    - Retrieving the relevant text chunks based on similarity search.
    - Fetching the exact content of an article using its article number.

Usage Example:
    >>> from models.rag_model import RAGModel
    >>> rag_model = RAGModel()
    >>> parsed_code = rag_model.parse_penal_code(legal_text)
    >>> chunks = rag_model.chunk_text(legal_text)
    >>> embeddings = rag_model.generate_embeddings()
    >>> rag_model.store_embeddings()
    >>> rag_model.load_embeddings()
    >>> relevant_chunks = rag_model.get_relevant_chunks("query text")
    >>> article_content = rag_model.get_exact_article("5.2.1")

Dependencies:
    - langchain_openai: Provides the OpenAIEmbeddings class for generating text embeddings.
    - faiss: Facilitates storage and similarity search of embeddings.
    - numpy: Used for numerical operations and array manipulations.
    - re: Utilized for parsing and pattern matching in legal texts.
    - config: Supplies configuration settings such as the OpenAI API key.
"""

import logging

from llama_index.core.llms import ChatMessage

from config import Config
from utils.api import make_llm, set_openai_api_key
from utils.citation_handler import get_direct_citation, is_direct_citation_request
from utils.summarizer import summarize_history

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

    # pylint: disable=too-many-locals, too-many-statements

    set_openai_api_key()

    logger.info("Received query:\n%s", query)

    if conversation_history is None:
        conversation_history = []

    conversation_history = [
        ChatMessage(**msg) if isinstance(msg, dict) else msg
        for msg in conversation_history
    ]

    conversation_history.append(ChatMessage(role="user", content=query))

    total_estimated_cost = 0.0  # Initialize the total cost variable

    # Check if the query requests a direct citation
    if is_direct_citation_request(query):
        logger.info("Query requests specific section text")
        # Directly retrieve the citation from the RAG model
        answer = get_direct_citation(query, rag_model, conversation_history)
    else:
        # Summarize the conversation history
        summary, summary_usage = summarize_history(conversation_history)
        logger.info("Summarized conversation history for RAG model query:\n%s", summary)

        # Calculate cost for summarization
        if summary_usage:
            input_token_price = 0.15 / 1000000
            output_token_price = 0.60 / 1000000

            prompt_tokens = summary_usage.prompt_tokens
            completion_tokens = summary_usage.completion_tokens

            summary_cost = (prompt_tokens * input_token_price) + (
                completion_tokens * output_token_price
            )

            total_estimated_cost += summary_cost  # Add to total cost

            logger.info(
                "Summarization Token usage - Prompt: %d, Completion: %d",
                prompt_tokens,
                completion_tokens,
            )
            logger.info("Estimated cost of summarization: $%.5f", summary_cost)

        # Retrieve relevant chunks from the RAG model using the summary
        relevant_chunks, chunk_cost = rag_model.get_relevant_chunks(
            summary, k=Config.TOP_K
        )
        total_estimated_cost += chunk_cost

        if not relevant_chunks:
            # If no relevant chunks are found, return a message indicating that the answer
            # cannot be found
            answer = "Het antwoord op deze vraag kan niet worden gevonden in de gegeven context."

        else:
            context = "\n\n".join(relevant_chunks)
            print(context)
            # Prepare the messages for the LLM
            messages = [
                ChatMessage(role="system", content=Config.SYSTEM_ROLE),
                ChatMessage(role="system", content=f"Relevante context:\n{context}"),
            ] + conversation_history

            logger.info("Prepared messages for LLM:")
            for msg in messages:
                logger.info("%s: %s", msg.role, msg.content[:200])

            # Initialize the chat model (NIM or OpenAI)
            llm = make_llm(
                temperature=0,
                model=Config.CHAT_MODEL,
                max_tokens=Config.MAX_TOKENS,
            )

            # Get the response from the chat model
            responses = llm.chat(messages)
            answer = responses.message.content

            # Estimate and log the cost
            # The SDK always defines .usage, setting it to None when the endpoint
            # omits it, so hasattr() is always True and guards nothing.
            if getattr(responses.raw, "usage", None) is not None:
                token_usage = responses.raw.usage
                prompt_tokens = token_usage.prompt_tokens
                completion_tokens = token_usage.completion_tokens
                total_tokens = token_usage.total_tokens

                input_token_price = 0.15 / 1000000
                output_token_price = 0.60 / 1000000

                estimated_cost = (prompt_tokens * input_token_price) + (
                    completion_tokens * output_token_price
                )

                total_estimated_cost += estimated_cost  # Add to total cost

                logger.info(
                    "Token usage - Prompt: %d, Completion: %d, Total: %d",
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )
                logger.info("Estimated cost of this API call: $%.5f", estimated_cost)
            else:
                logger.warning("Token usage information is not available.")

    logger.info("Total estimated cost for the API calls: $%.5f", total_estimated_cost)
    logger.info("Generated answer:\n%s", answer)

    # Add the assistant's response to the conversation history
    conversation_history.append(ChatMessage(role="assistant", content=answer))

    # Convert the conversation history to a serializable format
    serializable_history = [
        {"role": msg.role, "content": msg.content} for msg in conversation_history
    ]

    return answer, serializable_history
