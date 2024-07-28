import sys
import os
import logging
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from init.init_rag_model import initialize_rag_model
from utils.answer_generator import generate_answer
from config import Config

app = Flask(__name__)

# Configuration for Flask-Session
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
Session(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for the RAG model
rag_model = None

@app.route("/")
def index():
    session.clear()
    logger.info("Session cleared and new session started")
    if Config.MOBILE:
        return render_template("index_mobile.html")
    else:
        return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    logger.info("Received user input: %s", user_input)

    conversation_history = session.get("conversation_history", [])
    response, updated_conversation_history = generate_answer(
        user_input, rag_model, conversation_history
    )

    logger.info("Generated response: %s", str(response)[:200])
    session["conversation_history"] = updated_conversation_history

    return jsonify({"response": response, "conversation_history": updated_conversation_history})

if __name__ == "__main__":
    # Initialize the RAG model
    rag_model = initialize_rag_model()
    app.run(host='0.0.0.0', port=5001, debug=True)  
