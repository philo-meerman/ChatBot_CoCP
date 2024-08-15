import unittest
from unittest.mock import patch, MagicMock
from utils.pdf_processor import extract_text_from_pdf, clean_text, extract_boek_2_text

class PDFProcessorTests(unittest.TestCase):

    @patch("utils.pdf_processor.PdfReader")
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        # Mock the PdfReader and its return value
        mock_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Mock the pages and their extract_text method
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 text. "
        mock_reader_instance.pages = [mock_page, mock_page]

        pdf_path = "dummy/path/to/pdf"
        text = extract_text_from_pdf(pdf_path)
        
        expected_text = "Page 1 text. Page 1 text. "
        self.assertEqual(text, expected_text)
        mock_pdf_reader.assert_called_once_with(pdf_path)

    def test_clean_text(self):
        text = "Some text. Tweede Kamer, vergaderjaar 2022–2023, 36 327, nr. 2 Some more text."
        cleaned_text = clean_text(text)
        
        expected_text = "Some text.  Some more text."
        self.assertEqual(cleaned_text, expected_text)

    @patch("utils.pdf_processor.extract_text_from_pdf")
    @patch("utils.pdf_processor.clean_text")
    def test_extract_boek_2_text(self, mock_clean_text, mock_extract_text_from_pdf):
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
    def test_extract_boek_2_text_start_marker_not_found(self, mock_extract_text_from_pdf):
        full_text = "Some irrelevant text."
        mock_extract_text_from_pdf.return_value = full_text

        pdf_path = "dummy/path/to/pdf"
        with self.assertRaises(ValueError) as context:
            extract_boek_2_text(pdf_path)

        self.assertTrue("Start marker 'BOEK 2' not found in the document." in str(context.exception))
        mock_extract_text_from_pdf.assert_called_once_with(pdf_path)

if __name__ == "__main__":
    unittest.main()
