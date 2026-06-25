"""
multi_resume_analyzer.py

Batch processes all resumes inside the `uploaded_resumes/` directory.
Uses the `resume_parser` module to extract text from PDFs and DOCX files.
Compiles everything into a single CSV dataset (`outputs/uploaded_resume_texts.csv`) 
ready to be ingested by the intelligent ranking pipeline.
"""

import os
import sys
import pandas as pd

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resume_upload.resume_parser import parse_resume

RESUME_FOLDER = os.path.join(PROJECT_ROOT, "uploaded_resumes")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "uploaded_resume_texts.csv")


def analyze_resumes():
    
    if not os.path.exists(RESUME_FOLDER):
        os.makedirs(RESUME_FOLDER)
        print(f"Created folder: {RESUME_FOLDER}")
        print("Please place PDF or DOCX resumes inside and run again.")
        return

    resumes = []
    
    # Get all pdf and docx files
    files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith(('.pdf', '.docx'))]
    
    if not files:
        print(f"No PDF or DOCX files found in {RESUME_FOLDER}")
        return
        
    print("=" * 60)
    print(f"MULTI-RESUME ANALYZER (Found {len(files)} files)")
    print("=" * 60)

    for file in files:
        path = os.path.join(RESUME_FOLDER, file)
        
        try:
            print(f"Processing: {file} ...", end=" ")
            text = parse_resume(path)
            
            resumes.append({
                "file_name": file,
                "resume_text": text
            })
            
            print(f"✅ ({len(text)} chars)")
            
        except Exception as e:
            print(f"❌ Skipped: {e}")

    # Compile the final dataset
    if resumes:
        df = pd.DataFrame(resumes)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n{'=' * 60}")
        print("✅ Batch resume text extraction completed!")
        print(f"Dataset compiled: {OUTPUT_FILE}")
        print(f"Total extracted: {len(df)} resumes")
    else:
        print("\n❌ Failed to extract text from any resumes.")


if __name__ == "__main__":
    analyze_resumes()
