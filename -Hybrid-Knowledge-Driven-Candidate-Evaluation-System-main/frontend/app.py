"""
app.py

Streamlit web interface for the AI Resume Analyzer.
Allows users to upload PDF/DOCX resumes, extracts their text, 
and runs the AI ranking pipeline to display scored candidates interactively.
"""

import streamlit as st
import os
import sys
import pandas as pd
import shutil

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resume_upload.resume_parser import parse_resume
from pipeline.integration_pipeline import run_full_pipeline

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploaded_resumes")
FINAL_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "ultimate_uploaded_rankings.csv")
TEXTS_FILE = os.path.join(PROJECT_ROOT, "outputs", "uploaded_resume_texts.csv")


def main():
    st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
    
    st.title("📄 AI Resume Analyzer")
    st.markdown("Upload candidate resumes (PDF/DOCX) to instantly analyze their suitability against the target Job Description.")

    # 1. Provide Upload Interface
    st.subheader("1. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resume Files (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Clear old resumes from the folder so we only analyze current session uploads
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        st.success(f"{len(uploaded_files)} resumes uploaded successfully! Ready for analysis.")


    # 2. Target Job Description
    st.subheader("2. Target Job Description")
    
    analysis_mode = st.radio(
        "Analysis Mode",
        options=["Multiple Candidate Screening Mode", "Single Resume Detailed Analysis Mode"],
        horizontal=True
    )
    
    st.markdown("#### Option A: Paste Custom Job Description")
    job_text_input = st.text_area("Paste Job Description (Automatically extracts required skills)", height=150)
    
    extracted_skills = []
    if job_text_input:
        from job_description.job_parser import extract_job_skills
        extracted_skills = extract_job_skills(job_text_input)
        
        if extracted_skills:
            st.info(f"**Detected Job Skills ({len(extracted_skills)}):** {', '.join(extracted_skills)}")
        else:
            st.warning("No known IT skills detected in the pasted text.")
            
        job_role_display = "Custom Job Description"
    else:
        st.markdown("#### Option B: Select Pre-defined Role")
        import json
        job_roles_path = os.path.join(PROJECT_ROOT, "knowledge_base", "job_roles.json")
        try:
            with open(job_roles_path, "r") as f:
                roles = json.load(f)
            role_options = list(roles.keys())
        except Exception:
            role_options = ["AI Engineer", "Data Scientist", "Software Engineer"]
            
        job_role = st.selectbox(
            "Select Target Job Role",
            role_options
        )
        job_role_display = job_role

    st.subheader("3. Analyze Candidates")
    if st.button(f"Analyze for {job_role_display}", type="primary"):
        
        # Check if there's anything to analyze
        if not uploaded_files:
            st.error("Please upload resumes first!")
            return
            
        with st.spinner("Running Full Hybrid AI Pipeline (ML + Logic + Graph)..."):
            try:
                # Instead of analyzing the whole folder, only parse these specific files
                resumes = []
                for file in uploaded_files:
                    file_path = os.path.join(UPLOAD_FOLDER, file.name)
                    text = parse_resume(file_path)
                    resumes.append({
                        "file_name": file.name,
                        "resume_text": text
                    })
                
                # Save just the currently uploaded resumes as our batch
                df = pd.DataFrame(resumes)
                df.to_csv(TEXTS_FILE, index=False)
                
                # NEW: Run the FULL HYBRID PIPELINE
                if job_text_input:
                    run_full_pipeline(target_role="Custom", custom_job_text=job_text_input, custom_job_skills=extracted_skills)
                else:
                    run_full_pipeline(target_role=job_role)
                
                st.success("Analysis completed!")
                
                # Load and display NEW FULL results
                if os.path.exists(FINAL_OUTPUT_FILE):
                    results = pd.read_csv(FINAL_OUTPUT_FILE)
                    
                    st.subheader(f"🏆 Candidate Rankings for {job_role_display}")
                    
                    # UI Presentation Mode
                    if analysis_mode == "Single Resume Detailed Analysis Mode" and len(results) == 1:
                        candidate_row = results.iloc[0]
                        st.info(f"**Candidate:** {candidate_row['Candidate']}")
                        
                        # Match Percentage Display (clamped to 0)
                        match_percentage = max(0, round(candidate_row['Final_Score'] * 100))
                        st.subheader(f"🎯 Candidate Match: {match_percentage}%")
                        
                        # Show beautiful metric breakdown
                        col1, col2, col3 = st.columns(3)
                        col1.metric("ML Text Match", f"{candidate_row['ML_Score']:.4f}")
                        col2.metric("Graph Expertise", f"{candidate_row['Graph_Score']:.4f}")
                        col3.metric("Rule Logic", f"{candidate_row['Rule_Score']:.4f}")
                        
                        st.markdown("#### Competency Analysis")
                        st.success(f"**✅ Matched Skills:**\n{candidate_row['Extracted_Skills']}")
                        
                        if "Missing_Skills" in candidate_row and candidate_row['Missing_Skills'] != "None":
                            st.error(f"**❌ Missing Skills:**\n{candidate_row['Missing_Skills']}")
                            
                        st.markdown(f"**⚡ Rules Triggered:** {candidate_row['Rules_Triggered']}")
                        
                        st.markdown("### Raw Ranking Data")
                        st.dataframe(
                            results.drop(columns=["Sections"], errors="ignore"),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Add Resume Section Visualizer
                        if "Sections" in candidate_row and pd.notna(candidate_row["Sections"]):
                            import json
                            st.markdown("---")
                            st.subheader("📑 Resume Section Analysis")
                            
                            try:
                                parsed_sections = json.loads(candidate_row["Sections"])
                                
                                # Use columns or expanders to make it look great
                                for sec_name, sec_text in parsed_sections.items():
                                    if sec_text:
                                        with st.expander(f"{sec_name.upper()}  ({len(sec_text)} chars)"):
                                            st.write(sec_text)
                            except Exception as e:
                                st.error(f"Could not parse section data: {e}")
                    else:
                        if analysis_mode == "Single Resume Detailed Analysis Mode":
                            st.warning("You selected Single Mode but uploaded multiple files. Displaying as Multiple Candidates Screening.")
                        
                        if "Final_Score" in results.columns:
                            # Clamp negative scores to 0%
                            raw_pct = (results["Final_Score"] * 100).round(0).clip(lower=0).astype(int)
                            results.insert(1, "Match_Percentage", raw_pct.astype(str) + "%")
                            
                        st.dataframe(
                            results.drop(columns=["Sections"], errors="ignore"),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.error("No results file generated. Check the terminal logs.")
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
