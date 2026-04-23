# 🤖 AI Resume Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Overview

**AI Resume Analyzer** is an intelligent tool that automates the resume screening process using Machine Learning. It analyzes resumes against job descriptions and provides match scores, skill gap analysis, and hiring recommendations.

### 🎯 Key Features

- 📄 **Multiple Format Support** - Upload PDF, DOCX, or TXT files
- 🤖 **ML-Powered Analysis** - Random Forest model trained on 9,500+ resumes
- 📊 **Match Score Prediction** - Get percentage match between resume and job
- 🔍 **Skill Extraction** - Automatically identifies technical and soft skills
- 💡 **Smart Recommendations** - "Strong Match", "Good Match", or "Poor Match"
- 📈 **Visual Analytics** - Interactive charts and graphs
- 📥 **Export Reports** - Download analysis results as CSV
- 📜 **Analysis History** - Track all previous analyses

### 🧠 How It Works

1. Upload a resume (PDF/DOCX/TXT)
2. Paste/upload job description
3. ML model analyzes:
   - Keyword similarity (TF-IDF)
   - Experience years extraction
   - Skills matching
   - Education level detection
4. Get instant match score + recommendations

### 📊 Model Performance

- **Algorithm**: Random Forest Regressor
- **Training Data**: 9,544 labeled resumes
- **Features**: 5 key parameters
- **R² Score**: 0.72
- **RMSE**: 0.12

### 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Backend | Python 3.8+ |
| ML Framework | Scikit-learn, Pandas, NumPy |
| Web Interface | Streamlit |
| PDF Processing | PyPDF2, pdfplumber |
| Visualization | Plotly |
| Model Export | Joblib |

### 📁 Project Structure
