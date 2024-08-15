# Chatbot Project

This project implements a chatbot using OpenAI's GPT-3.5 Turbo LLM and a RAG model for handling PDF files, with a browser-based user interface.

## Setup

1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Create a `.env` file in the project folder to store your `OPENAI_API_KEY` variable.
4. The Vector Store should be available. Adjust the Config `EMBED_REGEN` parameter to `1` to create it. Leave it `0` otherwise.
5. To run the tests, use the following command: `python -m unittest discover -s tests`.
6. Run the Flask app with `python web/app.py`.

## Usage

Access the chatbot interface at `http://127.0.0.1:5001/` in your browser.

## Explanation

- **data/**: Directory containing the PDF files.
- **evaluation/**: Directory of evaluation scripts, ground truth and comparison results.
- **models/**: Directory for RAG model-related scripts.
- **utils/**: Utility scripts for PDF processing and RAG model interaction.
- **web/**: Flask application files, including templates and static files.
- **tests/**: Unit tests for the application.
- **requirements.txt**: List of dependencies to be installed.
- **README.md**: This file.
