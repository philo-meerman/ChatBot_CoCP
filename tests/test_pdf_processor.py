"""
Unit Tests for the PDF Processor Module

This script contains a suite of unit tests that verify the functionality of the PDF processing
utilities provided in the PDF processor module. The tests are designed to validate the following:

- Extraction of text from PDF files using a mocked PdfReader, ensuring that text from each page is
  concatenated correctly.
- Cleaning of text to remove specific unwanted patterns, such as parliamentary document references.
- Extraction of the "Book 2" section from PDF content, confirming that the correct segment is
  retrieved and cleaned.
- Proper error handling when the expected "Book 2" start marker is not found in the extracted text.

Test Cases:
    - test_extract_text_from_pdf:
          Mocks the PdfReader to simulate text extraction from multiple pages and checks the
          concatenated output.
    - test_clean_text:
          Verifies that the clean_text function removes specified patterns while preserving the rest
          of the text.
    - test_extract_boek_2_text:
          Ensures that the extract_boek_2_text function correctly identifies and processes the "Book
          2" section.
    - test_extract_boek_2_text_start_marker_not_found:
          Checks that a ValueError is raised with an appropriate message when the "Book 2" marker is
          missing.

Usage:
    Run this test suite using Python's unittest framework:
        python -m unittest tests/test_pdf_processor.py

Dependencies:
    - unittest: Python's built-in testing framework.
    - unittest.mock: Used for mocking external dependencies in the PDF processing functions.
    - utils.pdf_processor: The module containing the PDF processing utilities under test.
"""

import unittest
from unittest.mock import patch, MagicMock
from utils.pdf_processor import extract_text_from_pdf, clean_text, extract_boek_2_text


class PDFProcessorTests(unittest.TestCase):
    """Unit tests for the PDF processor module.

    This test suite verifies the functionality of the PDF processing utilities, including
    text extraction from PDFs, text cleaning, and specific book section extraction.

    Test Cases:
        - test_extract_text_from_pdf: Tests the extraction of text from a PDF file
          using mocked PdfReader
        - test_clean_text: Verifies the text cleaning functionality, especially for
          parliamentary document references
        - test_extract_boek_2_text: Tests the extraction of Book 2 section from PDF content
        - test_extract_boek_2_text_start_marker_not_found: Verifies error handling when
          Book 2 marker is not found

    The tests use unittest.mock to mock external dependencies and verify the behavior
    of the PDF processing functions in isolation.
    """

    @patch("utils.pdf_processor.PdfReader")
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        """
        Test the extract_text_from_pdf function by mocking PdfReader.

        This test method verifies that:
        - The PdfReader is correctly instantiated with the given path
        - Text extraction is performed for each page
        - Extracted text from all pages is properly concatenated

        Args:
            mock_pdf_reader (MagicMock): A mock object for PyPDF2.PdfReader class

        Returns:
            None

        Raises:
            AssertionError: If the extracted text doesn't match the expected text or if
                           PdfReader is not called correctly with the given path
        """
        # Mock the PdfReader and its return value
        mock_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_reader_instance

        # Mock the pages and their extract_text method
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 text. "
        mock_reader_instance.pages = [mock_page, mock_page]

        pdf_path = "dummy/path/to/pdf"
        text = extract_text_from_pdf(pdf_path)

        # extract_text_from_pdf separates pages with "\n ", per its docstring.
        expected_text = "Page 1 text. \n Page 1 text. \n "
        self.assertEqual(text, expected_text)
        mock_pdf_reader.assert_called_once_with(pdf_path)

    def test_clean_text(self):
        """Test the clean_text function.

        Tests if the clean_text function correctly removes specific text patterns
        (like parliamentary document references) while preserving the rest of the text.

        Args:
            self: TestCase instance

        Expected:
            The function should remove the text "Tweede Kamer, vergaderjaar 2022–2023,
            36 327, nr. 2"
            while keeping the surrounding text intact and properly spaced.
        """
        text = "Some text. Tweede Kamer, vergaderjaar 2022–2023, 36 327, nr. 2 Some more text."
        cleaned_text = clean_text(text)

        expected_text = "Some text.  Some more text."
        self.assertEqual(cleaned_text, expected_text)

    @patch("utils.pdf_processor.extract_text_from_pdf")
    @patch("utils.pdf_processor.clean_text")
    def test_extract_boek_2_text(self, mock_clean_text, mock_extract_text_from_pdf):
        """Tests the extract_boek_2_text function.

        This test verifies that the function correctly extracts text from Book 2
        (BOEK 2 HET OPSPORINGSONDERZOEK) from a PDF document, including handling of text cleaning.

        Args:
            mock_clean_text (MagicMock): Mock for the clean_text function
            mock_extract_text_from_pdf (MagicMock): Mock for the extract_text_from_pdf function

        Returns:
            None

        Raises:
            AssertionError: If the extracted text doesn't match the expected output or if the mocks
                           are not called as expected
        """
        full_text = (
            "Some irrelevant text. "
            "BOEK 2 HET OPSPORINGSONDERZOEK Relevant text in Book 2. "
            "BOEK 3 BESLISSINGEN OVER VERVOLGING Some irrelevant text."
        )
        mock_extract_text_from_pdf.return_value = full_text
        mock_clean_text.side_effect = lambda x: x  # No-op for clean_text

        pdf_path = "dummy/path/to/pdf"
        boek_2_text = extract_boek_2_text(pdf_path)

        expected_text = "BOEK 2 HET OPSPORINGSONDERZOEK Relevant text in Book 2. "
        self.assertEqual(boek_2_text, expected_text)
        mock_extract_text_from_pdf.assert_called_once_with(pdf_path)
        mock_clean_text.assert_called_once_with(expected_text)

    @patch("utils.pdf_processor.extract_text_from_pdf")
    def test_extract_boek_2_text_start_marker_not_found(
        self, mock_extract_text_from_pdf
    ):
        """
        Test the behavior of extract_boek_2_text when the 'BOEK 2' marker is not found in the text.

        This test verifies that the function raises a ValueError with an appropriate error message
        when the text extracted from the PDF does not contain the required 'BOEK 2' start marker.

        Args:
            mock_extract_text_from_pdf: A mock object that simulates the PDF text extraction
            function

        Raises:
            ValueError: Expected to be raised when 'BOEK 2' marker is not found

        Asserts:
            - Verifies that the correct error message is included in the ValueError
            - Confirms that the mock extraction function was called exactly once with the correct
              path
        """
        full_text = "Some irrelevant text."
        mock_extract_text_from_pdf.return_value = full_text

        pdf_path = "dummy/path/to/pdf"
        with self.assertRaises(ValueError) as context:
            extract_boek_2_text(pdf_path)

        self.assertTrue(
            "Start marker 'BOEK 2' not found in the document." in str(context.exception)
        )
        mock_extract_text_from_pdf.assert_called_once_with(pdf_path)


if __name__ == "__main__":
    unittest.main()
