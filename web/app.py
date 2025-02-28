"""
Web Application for ChatBot_DPC

This module sets up a Flask-based web application for the ChatBot_DPC project.
It configures session management using Flask-Session, establishes logging, and defines
routes to interact with the Retrieval Augmented Generation (RAG) model.

Endpoints:
    - "/" (index):
          Clears the session and renders the appropriate template based on the user's device type,
          determined via the User-Agent header.
    - "/chat" (POST):
          Receives a user's message, processes it through the RAG model (using generate_answer),
          updates the conversation history stored in the session, and returns the generated response
          as JSON.

Dependencies:
    - Flask: Provides the web framework for routing and request handling.
    - Flask-Session: Manages session data storage.
    - user_agents: Detects device types from User-Agent strings.
    - init.init_rag_model: Contains the initialize_rag_model() function to initialize the RAG model.
    - utils.answer_generator: Contains the generate_answer()
        function to process user input with the RAG model.

Usage:
    Run this script directly to start the web server:
        $ python web/app.py

The application will initialize the RAG model and listen on port 5001.
"""

import sys
import os
import logging
from flask import Flask, render_template, request, jsonify, session
from user_agents import parse
from flask_session import Session  # pylint: disable=no-name-in-module

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: disable=wrong-import-position
from init.init_rag_model import initialize_rag_model
from utils.answer_generator import generate_answer

app = Flask(__name__)

# Configuration for Flask-Session
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
Session(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for the RAG model
RAG_MODEL = None


@app.route("/")
def index():
    """
    Handles the index route of the web application.

    This function clears any existing session data and serves the appropriate template
    based on the user's device type (mobile or desktop). It logs the user agent information
    and device detection results.

    Returns:
        template: Rendered HTML template ('index_mobile.html' for mobile devices,
                 'index.html' for desktop devices)
    """
    session.clear()
    logger.info("Session cleared and new session started")

    # Detect device type
    user_agent = request.headers.get("User-Agent")
    ua = parse(user_agent)

    logger.info("User-Agent: %s", user_agent)
    logger.info("Is Mobile: %s", ua.is_mobile)

    if ua.is_mobile:
        return render_template("index_mobile.html")
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat interactions between user and the RAG-enhanced model.

    This function processes incoming chat messages, maintains conversation history,
    and generates responses using the RAG model.

    Returns:
        JSON response containing:
            - response (str): The generated answer from the model
            - conversation_history (list): Updated conversation history

    Raises:
        Potential exceptions from request.json access or generate_answer function

    Note:
        - Requires active session management
        - Logs both incoming messages and generated responses
        - Conversation history is maintained in session
    """
    user_input = request.json.get("message")
    logger.info("Received user input: %s", user_input)

    conversation_history = session.get("conversation_history", [])
    response, updated_conversation_history = generate_answer(
        user_input, RAG_MODEL, conversation_history
    )

    logger.info("Generated response: %s", str(response)[:200])
    session["conversation_history"] = updated_conversation_history

    return jsonify(
        {"response": response, "conversation_history": updated_conversation_history}
    )


if __name__ == "__main__":
    # Initialize the RAG model
    RAG_MODEL = initialize_rag_model()
    app.run(host="0.0.0.0", port=5001, debug=True)
