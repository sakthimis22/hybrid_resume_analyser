"""
sample_test.py

A quick script to test the resume_parser module on a sample PDF.
"""

import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resume_upload.resume_parser import parse_resume


def create_mock_pdf(filename):
    """Create a temporary dummy PDF to test the parser if one doesn't exist."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="John Doe - Artificial Intelligence Engineer", ln=1, align="C")
    pdf.cell(200, 10, txt="", ln=1)
    
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, txt="Skills:", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Python, TensorFlow, PyTorch, React, Docker", ln=1)
    
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, txt="Experience: 5 years", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Senior AI Developer at TechCorp - Built hybrid ML ranking engine.", ln=1)
    
    pdf.output(filename)
    print(f"Created dummy test resume: {filename}")


if __name__ == "__main__":
    
    print("=" * 60)
    print("RESUME PARSER MODULE TEST")
    print("=" * 60)
    
    test_pdf = os.path.join(SCRIPT_DIR, "test_resume.pdf")
    
    # Generate mock PDF if it doesn't exist
    if not os.path.exists(test_pdf):
        create_mock_pdf(test_pdf)
        
    print(f"\nParsing {test_pdf}...")
    
    try:
        text = parse_resume(test_pdf)
        print("\n--- Extracted Text ---")
        print(text[:1000])
        print("----------------------\n")
        print(f"SUCCESS: Extracted {len(text)} characters of text from PDF.")
    except Exception as e:
        print(f"FAILED to parse: {e}")
