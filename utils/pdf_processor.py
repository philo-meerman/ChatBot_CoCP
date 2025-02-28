"""
PDF Processor Module

This module provides utility functions for extracting and cleaning text from PDF documents.
It is designed to support the ChatBot_DPC project by processing legal or parliamentary PDFs,
extracting relevant sections, and removing unwanted footer patterns.

Functions:
    extract_text_from_pdf(pdf_path):
        Reads a PDF file and concatenates the text from all its pages into a single string,
        separating pages with newlines.

    clean_text(text):
        Removes specific footer patterns (e.g., a Dutch parliament document footer) from the input
        text using regular expressions.

    extract_boek_2_text(pdf_path):
        Extracts the text content of "Book 2" from a PDF document. This function locates the section
        between the markers "BOEK 2 HET OPSPORINGSONDERZOEK" and "BOEK 3 BESLISSINGEN OVER
        VERVOLGING" (or until the end of the document if the end marker is not found), cleans the
        extracted text, and returns it.

Usage:
    >>> from utils.pdf_processor import extract_boek_2_text
    >>> text = extract_boek_2_text('path/to/document.pdf')
"""

import re
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.

    This function reads a PDF file and concatenates the text content from all pages
    into a single string, with pages separated by newlines.

    Args:
        pdf_path (str): The file path to the PDF document to be processed

    Returns:
        str: The extracted text content from all pages of the PDF document,
             with pages separated by newlines and spaces

    Raises:
        FileNotFoundError: If the specified PDF file does not exist
        PdfReadError: If the PDF file is invalid or cannot be read
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n "
    return text


def clean_text(text):
    """
    Remove specific footer patterns from text.

    This function removes the Dutch parliament document footer pattern from the input text
    using regular expressions.

    Args:
        text (str): The input text containing potential footer patterns to be removed.

    Returns:
        str: The cleaned text with footer patterns removed.

    Example:
        >>> text = "Some content\\nTweede Kamer, vergaderjaar 2022–2023, 36 327, nr. 2\\n
                    More content"
        >>> clean_text(text)
        'Some content\\nMore content'
    """
    footer_pattern = re.compile(
        r"Tweede Kamer, vergaderjaar 2022–2023, 36 327, nr\. 2", re.IGNORECASE
    )
    text = re.sub(footer_pattern, "", text)

    return text


def extract_boek_2_text(pdf_path):
    """
    Extracts the text content of Book 2 ('BOEK 2') from a PDF document.

    This function extracts text between the markers 'BOEK 2 HET OPSPORINGSONDERZOEK' and
    'BOEK 3 BESLISSINGEN OVER VERVOLGING' from a given PDF file. If the end marker is not found,
    it extracts until the end of the document.

    Args:
        pdf_path (str): The file path to the PDF document to process.

    Returns:
        str: The cleaned text content of Book 2 from the PDF.

    Raises:
        ValueError: If the start marker 'BOEK 2' is not found in the document.
    """
    full_text = extract_text_from_pdf(pdf_path)

    start_marker = "BOEK 2 HET OPSPORINGSONDERZOEK"
    end_marker = "BOEK 3 BESLISSINGEN OVER VERVOLGING"

    start_pos = full_text.find(start_marker)
    if start_pos == -1:
        raise ValueError("Start marker 'BOEK 2' not found in the document.")

    end_pos = full_text.find(end_marker, start_pos)
    if end_pos == -1:
        end_pos = len(full_text)

    boek_2_text = clean_text(full_text[start_pos:end_pos])

    return boek_2_text
