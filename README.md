# AI Resume & Job Matching Platform

An AI-powered resume analysis tool that compares a candidate's resume with a job description using semantic embeddings and LLM-based analysis.

The application goes beyond simple keyword matching by identifying semantic similarity, relevant skills, missing keywords, experience gaps, and actionable resume improvement suggestions.

## Live Demo

[Launch the AI Resume Matcher](https://ai-resume-matcher-saow9iggthchyc7djblovz.streamlit.app/)

## Overview

The AI Resume & Job Matching Platform helps candidates understand how closely their resume aligns with a specific job description.

Instead of relying only on exact keyword overlap, the application uses embeddings-based semantic similarity to compare the meaning of the resume and job description.

It also uses an LLM to analyze the job requirements and provide targeted feedback.

## Key Features

- Upload a resume in PDF format
- Extract resume text automatically
- Calculate an embeddings-based semantic match score from 0–100
- Identify matched keywords and skills
- Identify missing keywords and skills
- Detect potential experience gaps when a role explicitly requires prior experience
- Warn when the provided job description is too short for a reliable comparison
- Generate specific, actionable resume improvement suggestions
- Responsive Streamlit interface for desktop and mobile screens

## How It Works

```text
Resume PDF
    ↓
PDF Text Extraction
    ↓
Resume + Job Description
    ↓
Semantic Embeddings
    ↓
Match Score
    ↓
LLM Analysis
    ↓
Matched Skills
Missing Skills
Experience Gap
Resume Suggestions
```

## Semantic Matching

The application uses Cohere embeddings to compare the semantic meaning of the resume and job description.

This means the system can recognize related concepts even when the exact wording differs.

For example:

```text
"Led a team of 5"
        ↓
"People management experience"
```

These concepts can be recognized as semantically related even without identical wording.

## AI-Powered Resume Analysis

After calculating the semantic match score, the application analyzes the resume and job description to identify:

### Matched Keywords

Important skills or terms from the job description that are represented in the resume.

### Missing Keywords

Important job requirements that are not reflected in the resume.

### Experience Gap

If the job description explicitly requires prior professional experience and the resume does not demonstrate comparable experience, the application highlights this as a potential gap.

### Resume Suggestions

The application provides specific suggestions for better presenting and tailoring experience that already exists in the resume.

## Tech Stack

- **Python**
- **Streamlit**
- **Cohere API**
  - Embeddings
  - LLM analysis
- **pypdf**
- **NumPy**

## Project Structure

```text
AI-Resume-Matcher/
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The application requires a Cohere API key.

For local development, you can provide the key through Streamlit secrets or enter it through the application's sidebar.

Example Streamlit secrets configuration:

```toml
COHERE_API_KEY = "your_api_key_here"
```

**Never commit API keys or other secrets to GitHub.**

## Run Locally

```bash
streamlit run app.py
```

The application will open in your browser.

## Usage

1. Upload your resume as a PDF.
2. Paste the complete job description.
3. Click **Analyze Match**.
4. Review the overall match score.
5. Review matched and missing keywords.
6. Check for any experience-gap warning.
7. Read the AI-generated resume improvement suggestions.

## Limitations

- The quality of the analysis depends on the amount and quality of text available in the resume and job description.
- Scanned/image-only PDFs may not provide extractable text.
- A very short job description may produce a less reliable comparison.
- Semantic similarity is an estimate and should be used as a supporting signal rather than a definitive hiring decision.

## Future Improvements

Potential future enhancements include:

- Support for additional resume formats
- More detailed skill categorization
- Improved evaluation and benchmarking of match scores
- Resume section-level analysis
- Additional job-specific analytics

## Author

**Hammesh**

Built as an AI/ML portfolio project focused on NLP, semantic embeddings, and practical LLM-powered applications.
