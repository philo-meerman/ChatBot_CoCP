# Chatbot Project

This project implements a chatbot using a Huggingface LLM and a RAG model for handling PDF files, with a browser-based user interface.

## Setup

1. Clone the repository. 
2. Install dependencies with `pip install -r requirements.txt`.
3. To run the tests, use the following command: `python -m unittest discover -s tests`.
4. Run the Flask app with `python web/app.py`.

## Usage

Access the chatbot interface at `http://127.0.0.1:5000/` in your browser.

## Explanation
data/: Directory containing the PDF files.
models/: Directory for model-related scripts.
utils/: Utility scripts for PDF processing and RAG model interaction.
web/: Flask application files, including templates and static files.
tests/: Unit tests for the application.
requirements.txt: List of dependencies to be installed.
README.md: This file.