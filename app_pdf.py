# filename: app_pdf.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import base64
from datetime import datetime
import io
import PyPDF2
import pdfplumber
from docx import Document
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer - PDF Support",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .match-excellent {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .match-good {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .match-poor {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('resume_analyzer_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        return model, tfidf
    except FileNotFoundError:
        st.warning("⚠️ Model files not found! Using rule-based scoring (no ML).")
        return None, None

def extract_text_from_pdf(file):
    """Extract text from PDF file using multiple methods"""
    text = ""
    
    # Method 1: Try PyPDF2
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"PyPDF2 extraction issue: {e}")
    
    # If PyPDF2 didn't get much text, try pdfplumber (better for complex PDFs)
    if len(text.strip()) < 100:
        try:
            file.seek(0)  # Reset file pointer
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.warning(f"pdfplumber extraction issue: {e}")
    
    return text.strip() if text.strip() else "Could not extract text from PDF. Ensure it's not a scanned image."

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip() if text.strip() else "No text found in DOCX"
    except Exception as e:
        return f"Error extracting text: {e}"

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    try:
        text = file.read().decode('utf-8')
        return text.strip() if text.strip() else "No text found in file"
    except Exception as e:
        return f"Error extracting text: {e}"

def extract_text_from_file(uploaded_file):
    """Route file to appropriate extractor based on type"""
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    else:
        return f"Unsupported file type: {file_type}. Please upload PDF, DOCX, or TXT."

def extract_experience_years(text):
    """Extract years of experience from text"""
    if not text:
        return 0
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s+of\s+experience',
        r'(\d+)\+?\s*ye?a?r?s?',
        r'(\d+)\+?\s*yrs',
        r'experience of (\d+)',
        r'(\d+)\+? years? of experience',
        r'(\d+)\+? yr',
        r'(\d+)\s*\+\s*years?'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            years = []
            for m in matches:
                try:
                    years.append(int(m))
                except:
                    pass
            if years:
                return max(years)
    
    # Look for date ranges as fallback
    date_patterns = [
        r'(20\d{2})\s*[-–to]+\s*(?:20\d{2}|present)',
        r'(19\d{2})\s*[-–to]+\s*(?:19\d{2}|20\d{2}|present)'
    ]
    total_years = 0
    for pattern in date_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                start = int(match)
                total_years += 1
            except:
                pass
    if total_years > 0:
        return min(total_years, 30)
    
    return 0

def extract_skills_from_resume(text):
    """Extract common tech skills from resume"""
    common_skills = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular',
        'vue', 'node', 'django', 'flask', 'spring', 'c++', 'c#', 'php', 'ruby',
        'swift', 'kotlin', 'go', 'rust', 'typescript', 'mongodb', 'postgresql',
        'mysql', 'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'tableau',
        'power bi', 'excel', 'machine learning', 'deep learning', 'nlp',
        'computer vision', 'data science', 'data analysis', 'data engineering',
        'devops', 'agile', 'scrum', 'git', 'jenkins', 'ci/cd'
    ]
    
    text_lower = text.lower()
    found_skills = [skill for skill in common_skills if skill in text_lower]
    return found_skills

def analyze_resume_ml(model, tfidf, resume_text, job_description):
    """Predict match score using ML model"""
    try:
        resume_vec = tfidf.transform([resume_text.lower()])
        job_vec = tfidf.transform([job_description.lower()])
        sim_score = cosine_similarity(resume_vec, job_vec)[0][0]
        exp_years = extract_experience_years(resume_text)
        resume_len = len(resume_text.split())
        
        features = np.array([[sim_score, exp_years, 0, 0, resume_len]])
        predicted_score = model.predict(features)[0]
        predicted_score = max(0, min(1, predicted_score))
        
        return predicted_score, sim_score
    except:
        return None, None

def analyze_resume_rule_based(resume_text, job_description):
    """Fallback rule-based scoring when ML model is not available"""
    resume_lower = resume_text.lower()
    job_lower = job_description.lower()
    
    # Keyword matching
    resume_words = set(re.findall(r'\b\w+\b', resume_lower))
    job_words = set(re.findall(r'\b\w+\b', job_lower))
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'you',
                'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'}
    
    resume_keywords = {w for w in resume_words if w not in stopwords and len(w) > 2}
    job_keywords = {w for w in job_words if w not in stopwords and len(w) > 2}
    
    if not job_keywords:
        return 0.5, 0.3
    
    # Calculate similarity
    common = len(resume_keywords.intersection(job_keywords))
    total = len(job_keywords)
    keyword_score = common / total if total > 0 else 0
    
    # Experience boost
    exp_years = extract_experience_years(resume_text)
    exp_score = min(exp_years / 5, 0.2)  # max 0.2 boost for 5+ years
    
    # Length penalty (too short or too long)
    length = len(resume_text.split())
    if length < 100:
        length_score = -0.1
    elif length > 1500:
        length_score = -0.05
    else:
        length_score = 0
    
    total_score = keyword_score + exp_score + length_score
    total_score = max(0, min(1, total_score))
    
    # Simple similarity (for display)
    sim_score = keyword_score
    
    return total_score, sim_score

def create_gauge_chart(score, title="Match Score"):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': title, 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 30], 'color': '#ff6b6b'},
                {'range': [30, 50], 'color': '#ffd93d'},
                {'range': [50, 70], 'color': '#6bcf7f'},
                {'range': [70, 100], 'color': '#1e8e3e'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_recommendation(score):
    """Get recommendation based on score"""
    if score >= 70:
        return {
            'text': "⭐⭐⭐ STRONG RECOMMENDATION",
            'color': "match-excellent",
            'message': "This candidate is an excellent match! Strongly recommend moving forward with interview."
        }
    elif score >= 50:
        return {
            'text': "⭐⭐ GOOD MATCH",
            'color': "match-good",
            'message': "This candidate shows good potential. Consider interviewing to explore further."
        }
    else:
        return {
            'text': "⭐ POOR MATCH",
            'color': "match-poor",
            'message': "This candidate may not be suitable for this role. Continue searching."
        }

# ============================================
# MAIN APP
# ============================================

def main():
    st.markdown('<div class="main-header">📄 AI Resume Analyzer (PDF Support)</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/pdf.png", width=80)
        st.markdown("## 📋 About")
        st.markdown("""
        **AI-powered Resume Analyzer** that evaluates resumes against job descriptions.
        
        ### Supported Formats:
        - 📄 **PDF** (including scanned/text)
        - 📝 **DOCX** (Word documents)
        - 📃 **TXT** (Text files)
        
        ### Features:
        - 📊 Match score calculation
        - 🔍 Keyword analysis
        - 💼 Experience extraction
        - 🎯 Skills identification
        - 💡 Recommendations
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        1. Use **standard fonts** in PDFs for best extraction
        2. Include **years of experience** explicitly
        3. List **technical skills** clearly
        4. Use **standard section headings**
        """)
        
        # Model status
        st.markdown("---")
        if model is not None:
            st.success("✅ ML Model: Active")
        else:
            st.info("ℹ️ ML Model: Not available (using rule-based scoring)")
    
    # Main content - Two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 📄 Upload Resume")
        
        # File upload for resume
        resume_file = st.file_uploader(
            "Choose file (PDF, DOCX, or TXT)",
            type=['pdf', 'docx', 'txt'],
            key="resume",
            help="Upload resume in PDF, DOCX, or TXT format"
        )
        
        # Or paste text
        resume_text = st.text_area(
            "Or paste resume text directly:",
            height=200,
            placeholder="Paste your resume text here...",
            key="resume_text"
        )
        
        if resume_file:
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_file(resume_file)
                if extracted_text and "Error" not in extracted_text and "Could not extract" not in extracted_text:
                    resume_text = extracted_text
                    st.success(f"✅ Successfully extracted {len(resume_text.split())} words from {resume_file.name}")
                    
                    # Show preview
                    with st.expander("📖 Extracted Text Preview"):
                        st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
                else:
                    st.error(f"❌ {extracted_text}")
    
    with col2:
        st.markdown("## 💼 Job Description")
        
        # File upload for job description
        job_file = st.file_uploader(
            "Choose file (PDF, DOCX, or TXT)",
            type=['pdf', 'docx', 'txt'],
            key="job",
            help="Upload job description in PDF, DOCX, or TXT format"
        )
        
        # Or paste text
        job_text = st.text_area(
            "Or paste job description directly:",
            height=200,
            placeholder="Paste job description here...",
            key="job_text"
        )
        
        if job_file:
            with st.spinner("Extracting text from file..."):
                extracted_text = extract_text_from_file(job_file)
                if extracted_text and "Error" not in extracted_text and "Could not extract" not in extracted_text:
                    job_text = extracted_text
                    st.success(f"✅ Successfully extracted {len(job_text.split())} words from {job_file.name}")
                    
                    with st.expander("📖 Extracted Text Preview"):
                        st.text(job_text[:500] + "..." if len(job_text) > 500 else job_text)
                else:
                    st.error(f"❌ {extracted_text}")
    
    # Analyze button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("🔍 ANALYZE MATCH", use_container_width=True)
    
    if analyze_clicked:
        if not resume_text.strip():
            st.error("❌ Please upload or paste a resume!")
        elif not job_text.strip():
            st.error("❌ Please upload or paste a job description!")
        else:
            with st.spinner("Analyzing resume against job description... 🔄"):
                # Use ML if available, else rule-based
                if model is not None and tfidf is not None:
                    match_score, similarity_score = analyze_resume_ml(model, tfidf, resume_text, job_text)
                    analysis_type = "Machine Learning"
                else:
                    match_score, similarity_score = analyze_resume_rule_based(resume_text, job_text)
                    analysis_type = "Rule-Based"
                
                match_score_pct = match_score * 100
                similarity_score_pct = similarity_score * 100
                
                # Extract additional info
                exp_years = extract_experience_years(resume_text)
                skills = extract_skills_from_resume(resume_text)
                resume_word_count = len(resume_text.split())
                job_word_count = len(job_text.split())
                
                # Get recommendation
                rec = get_recommendation(match_score_pct)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                st.caption(f"Analysis method: {analysis_type}")
                
                # Score cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="{rec['color']}">
                        <h2>{match_score_pct:.1f}%</h2>
                        <p>Match Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="match-good">
                        <h2>{similarity_score_pct:.1f}%</h2>
                        <p>Content Similarity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="match-good">
                        <h2>{exp_years} years</h2>
                        <p>Experience Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="match-good">
                        <h2>{len(skills)}</h2>
                        <p>Skills Identified</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendation message
                st.markdown(f"""
                <div class="{rec['color']}" style="margin-top: 1rem;">
                    <h3>{rec['text']}</h3>
                    <p>{rec['message']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_gauge = create_gauge_chart(match_score_pct, "Match Score")
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Skills comparison
                    job_skills = extract_skills_from_resume(job_text)
                    common_skills = set(skills) & set(job_skills)
                    missing_skills = set(job_skills) - set(skills)
                    
                    if job_skills:
                        skill_data = pd.DataFrame({
                            'Category': ['Matched', 'Missing'],
                            'Count': [len(common_skills), len(missing_skills)]
                        })
                        fig_pie = px.pie(skill_data, values='Count', names='Category', 
                                        title='Skills Match Analysis',
                                        color_discrete_map={'Matched': '#28a745', 'Missing': '#dc3545'})
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                # Detailed analysis
                with st.expander("📋 Detailed Analysis", expanded=False):
                    st.markdown("### ✅ Matched Skills")
                    if common_skills:
                        st.write(", ".join(sorted(common_skills)))
                    else:
                        st.warning("No common skills found")
                    
                    st.markdown("### ❌ Missing Skills")
                    if missing_skills:
                        st.write(", ".join(sorted(missing_skills)))
                    else:
                        st.success("All required skills found!")
                    
                    st.markdown("### 📊 Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Resume Word Count", resume_word_count)
                        st.metric("Job Description Word Count", job_word_count)
                    with col2:
                        st.metric("Experience Years Found", exp_years)
                        st.metric("Total Skills Identified", len(skills))
                    
                    # Keyword clouds (simple text list)
                    st.markdown("### 🔑 Top Keywords in Resume")
                    words = re.findall(r'\b[a-z]{4,}\b', resume_text.lower())
                    word_freq = pd.Series(words).value_counts().head(20)
                    st.write(", ".join(word_freq.index.tolist()))
                
                # Save history
                if 'analysis_history' not in st.session_state:
                    st.session_state.analysis_history = []
                
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'match_score': match_score_pct,
                    'similarity': similarity_score_pct,
                    'experience': exp_years,
                    'skills': len(skills),
                    'resume_words': resume_word_count
                })
                
                # History
                if len(st.session_state.analysis_history) > 1:
                    with st.expander("📜 Previous Analyses", expanded=False):
                        history_df = pd.DataFrame(st.session_state.analysis_history)
                        st.dataframe(history_df, use_container_width=True)
                        
                        # Trend chart
                        fig_trend = px.line(history_df, x='timestamp', y='match_score',
                                           title='Match Score Trend')
                        st.plotly_chart(fig_trend, use_container_width=True)
                
                # Download report
                report = {
                    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Match Score (%)': match_score_pct,
                    'Similarity Score (%)': similarity_score_pct,
                    'Experience Years Found': exp_years,
                    'Skills Identified': len(skills),
                    'Skills List': ', '.join(skills),
                    'Matched Skills': ', '.join(common_skills),
                    'Missing Skills': ', '.join(missing_skills),
                    'Resume Word Count': resume_word_count,
                    'Job Word Count': job_word_count,
                    'Recommendation': rec['text']
                }
                
                report_df = pd.DataFrame([report])
                
                def get_download_link(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    return f'<a href="data:file/csv;base64,{b64}" download="resume_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 Download Full Report (CSV)</a>'
                
                st.markdown("---")
                st.markdown(get_download_link(report_df), unsafe_allow_html=True)

# ============================================
# RUN THE APP
# ============================================

if __name__ == "__main__":
    model, tfidf = load_model()
    main()