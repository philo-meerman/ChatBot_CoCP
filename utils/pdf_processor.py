# utils/pdf_processor.py
import re
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text


def clean_text(text):
    footer_pattern = re.compile(
        r"Tweede Kamer, vergaderjaar 2022–2023, 36 327, nr\. 2", re.IGNORECASE
    )
    text = re.sub(footer_pattern, "", text)

    lines = text.split("\n")
    joined_lines = []
    current_item = ""
    is_list_item = False

    for line in lines:
        if re.match(r"^\s*(-|\d+\.)\s+", line):
            if current_item:
                joined_lines.append(current_item.strip())
            current_item = line
            is_list_item = True
        else:
            if is_list_item:
                current_item += " " + line.strip()
            else:
                if current_item:
                    joined_lines.append(current_item.strip())
                current_item = line
                is_list_item = False

    if current_item:
        joined_lines.append(current_item.strip())

    cleaned_text = "\n".join(joined_lines)
    return cleaned_text


def extract_boek_2_text(pdf_path):
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


# def extract_boek_2_text(self, pdf_path):
#     full_text = self.extract_text_from_pdf(pdf_path)

#     start_marker = "BOEK 2"
#     end_marker = "BOEK 3"

#     start_pos = full_text.find(start_marker)
#     if start_pos == -1:
#         raise ValueError("Start marker 'BOEK 2' not found in the document.")

#     end_pos = full_text.find(end_marker, start_pos)
#     if end_pos == -1:
#         end_pos = len(full_text)

#     boek_2_text = full_text[start_pos:end_pos]
#     cleaned_text = self.clean_text(boek_2_text)
#     return cleaned_text
