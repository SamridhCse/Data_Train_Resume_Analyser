# filename: use_analyzer.py

import joblib
import numpy as np
import re

# Load the trained model
model = joblib.load('resume_analyzer_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def extract_experience_years(text):
    """Extract years of experience from text"""
    if not text:
        return 0
    patterns = [
        r'(\d+)\+?\s*ye?a?r?s?',
        r'(\d+)\+?\s*yrs',
        r'experience of (\d+)',
        r'(\d+)\+? years of experience'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            years = [int(m) for m in matches if m.isdigit()]
            if years:
                return max(years)
    return 0

def analyze_resume(resume_text, job_description):
    """
    Predict match score between a resume and job description
    
    Parameters:
    resume_text (str): Text content of the resume
    job_description (str): Text content of the job requirement
    
    Returns:
    dict: Prediction and analysis
    """
    try:
        # Convert texts to vectors
        resume_vec = tfidf.transform([resume_text.lower()])
        job_vec = tfidf.transform([job_description.lower()])
        
        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_score = cosine_similarity(resume_vec, job_vec)[0][0]
        
        # Extract features
        exp_years = extract_experience_years(resume_text)
        resume_len = len(resume_text.split())
        
        # Create feature array
        features = np.array([[sim_score, exp_years, 0, 0, resume_len]])
        
        # Predict
        predicted_score = model.predict(features)[0]
        
        # Determine recommendation
        if predicted_score >= 0.7:
            recommendation = "✅ Strong Match - Highly Recommended"
        elif predicted_score >= 0.5:
            recommendation = "⚠️ Moderate Match - Consider"
        else:
            recommendation = "❌ Weak Match - Not Recommended"
        
        return {
            'match_score': round(predicted_score * 100, 1),  # Percentage
            'similarity_score': round(sim_score * 100, 1),
            'extracted_experience': exp_years,
            'recommendation': recommendation,
            'confidence': 'High' if predicted_score > 0.7 else 'Medium' if predicted_score > 0.4 else 'Low'
        }
    except Exception as e:
        return {'error': str(e), 'match_score': 0}

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🤖 AI RESUME ANALYZER")
    print("="*60)
    
    # Example 1: Good match
    resume1 = """
    Python developer with 5 years of experience in machine learning and data science.
    Skilled in TensorFlow, PyTorch, and Scikit-learn. Master's degree in Computer Science.
    Worked on NLP projects and computer vision applications.
    """
    
    job1 = """
    Looking for a Machine Learning Engineer with Python skills, 
    experience in TensorFlow, and strong background in data science.
    Master's degree preferred.
    """
    
    print("\n📄 EXAMPLE 1: Good Match")
    print("-"*40)
    result1 = analyze_resume(resume1, job1)
    for key, value in result1.items():
        print(f"  {key}: {value}")
    
    # Example 2: Poor match
    resume2 = """
    Accountant with 3 years of experience in financial reporting and tax preparation.
    Skilled in QuickBooks and Excel. Bachelor's degree in Accounting.
    """
    
    job2 = """
    Looking for a Machine Learning Engineer with Python skills,
    experience in TensorFlow, and strong background in data science.
    """
    
    print("\n📄 EXAMPLE 2: Poor Match")
    print("-"*40)
    result2 = analyze_resume(resume2, job2)
    for key, value in result2.items():
        print(f"  {key}: {value}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("💡 INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  1. Analyze a new resume")
        print("  2. Exit")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '2':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            print("\n📝 Enter resume text (or type 'done' on a new line when finished):")
            resume_lines = []
            while True:
                line = input()
                if line.lower() == 'done':
                    break
                resume_lines.append(line)
            resume_text = ' '.join(resume_lines)
            
            print("\n💼 Enter job description (or type 'done' on a new line when finished):")
            job_lines = []
            while True:
                line = input()
                if line.lower() == 'done':
                    break
                job_lines.append(line)
            job_text = ' '.join(job_lines)
            
            if resume_text and job_text:
                result = analyze_resume(resume_text, job_text)
                print("\n" + "="*40)
                print("📊 ANALYSIS RESULT")
                print("="*40)
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print("\n⚠️ Please enter both resume and job description.")
        else:
            print("\n⚠️ Invalid choice. Enter 1 or 2.")