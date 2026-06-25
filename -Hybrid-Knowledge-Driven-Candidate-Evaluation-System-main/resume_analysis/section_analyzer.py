import re

SECTION_HEADERS = {
    "education": ["education", "academic background", "academic"],
    "experience": ["experience", "work experience", "employment", "professional experience", "history"],
    "projects": ["projects", "project experience", "academic projects"],
    "skills": ["skills", "technical skills", "core competencies"],
    "certifications": ["certifications", "certification", "licenses"]
}


def analyze_resume_sections(text):
    """
    Scans a raw resume string and structurally decomposes it into distinct sections
    based on common ATS header keywords.
    """
    text = text.lower()

    sections = {
        "education": "",
        "experience": "",
        "projects": "",
        "skills": "",
        "certifications": "",
        "objective": "" # Added an extra catch-all for top-level summaries
    }

    lines = text.split("\n")
    
    # Assume everything before the first section is the Objective/Summary
    current_section = "objective"

    for line in lines:
        line_clean = line.strip()
        
        # If it's empty, just append the newline and continue
        if not line_clean:
            if current_section:
                sections[current_section] += "\n"
            continue

        # Look for headers
        # We check if the line *is exactly* a keyword, 
        # or if it's very short and contains a keyword to avoid false positives mid-sentence
        header_found = False
        if len(line_clean) < 40:  # Headers are usually short lines
            for section, keywords in SECTION_HEADERS.items():
                if any(k in line_clean for k in keywords):
                    current_section = section
                    header_found = True
                    break

        # Don't append the actual header string to the section content
        if not header_found and current_section:
            sections[current_section] += line + "\n"

    # Clean up excess whitespace
    for k in sections:
        sections[k] = sections[k].strip()

    return sections


if __name__ == "__main__":
    # Quick visual test
    sample_text = """
    JOHN DOE
    A passionate AI Engineer seeking new opportunities.
    
    EDUCATION
    Integrated M.Tech Software Engineering
    VIT Vellore, 2026
    
    EXPERIENCE
    Cybersecurity Intern – Retech Solutions
    Built AI-based malware detection systems.
    
    PROJECTS
    Internet-Free Tracking System
    Indoor Waste Detection System
    
    SKILLS
    Python, Java, SQL, Machine Learning
    """
    
    res = analyze_resume_sections(sample_text)
    for k, v in res.items():
        if v:
            print(f"[{k.upper()}]")
            print(v)
            print("-" * 20)
