"""
resume_parser.py

Module to extract text from real PDF and DOCX resume files.
Converts documents into raw text strings that can be processed
by the intelligence pipeline (Skill Extraction, Inference, Ranking).
"""

import os
import pdfplumber
import docx


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    
    return text.strip()


def extract_text_from_docx(file_path):
    """
    Extract text from a DOCX file using python-docx.
    """
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        
    return text.strip()


def parse_resume(file_path):
    """
    Main entry point to parse a resume file based on its extension.
    Supported: .pdf, .docx
    
    Returns:
        str: The extracted text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Resume file not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only .pdf and .docx are supported.")


# ==================== QUICK TEST ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Parsing: {test_file}")
        try:
            content = parse_resume(test_file)
            print("-" * 60)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 60)
            print(f"Total characters extracted: {len(content)}")
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print("Usage: python resume_parser.py <path_to_resume.pdf|.docx>")
