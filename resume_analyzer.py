# filename: resume_analyzer.py

import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD THE CSV FILE (Handle BOM characters)
# ============================================

# Load with encoding that handles BOM
df = pd.read_csv('resume_data.csv', encoding='utf-8-sig')

print("Original dataset shape:", df.shape)
print("\nAll columns:")
for i, col in enumerate(df.columns):
    print(f"  {i}: '{col}'")

# Clean column names (remove any special characters, spaces, etc.)
df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
df.columns = df.columns.str.replace('\n', '', regex=False)
df.columns = df.columns.str.replace(' ', '_')

print("\nCleaned columns:")
for i, col in enumerate(df.columns):
    print(f"  {i}: '{col}'")

# ============================================
# STEP 2: FIND THE RIGHT COLUMN NAMES
# ============================================

# Find job position column (might have different names)
job_position_col = None
for col in df.columns:
    if 'job' in col.lower() and ('position' in col.lower() or 'name' in col.lower()):
        job_position_col = col
        break

# Find education requirement column
edu_req_col = None
for col in df.columns:
    if 'education' in col.lower() and 'requirement' in col.lower():
        edu_req_col = col
        break

# Find experience requirement column
exp_req_col = None
for col in df.columns:
    if 'experience' in col.lower() and 'requirement' in col.lower():
        exp_req_col = col
        break

# Find skills required column
skills_req_col = None
for col in df.columns:
    if 'skills_required' in col.lower():
        skills_req_col = col
        break

# Find responsibilities column
responsibilities_col = None
for col in df.columns:
    if 'responsibilities' in col.lower():
        responsibilities_col = col
        break

print(f"\n✓ Found job position column: {job_position_col}")
print(f"✓ Found education requirement column: {edu_req_col}")
print(f"✓ Found experience requirement column: {exp_req_col}")
print(f"✓ Found skills required column: {skills_req_col}")
print(f"✓ Found responsibilities column: {responsibilities_col}")

# ============================================
# STEP 3: CLEAN AND PREPARE DATA
# ============================================

def safe_literal_eval(x):
    """Safely convert stringified lists to actual lists"""
    if pd.isna(x):
        return []
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        if isinstance(x, str) and x.startswith('['):
            return ast.literal_eval(x)
        else:
            return [str(x)] if str(x).strip() else []
    except:
        return []

def safe_str(x):
    """Safely convert any value to string"""
    if pd.isna(x):
        return ""
    if x is None:
        return ""
    return str(x)

# Convert list columns safely
if 'skills' in df.columns:
    df['skills'] = df['skills'].apply(safe_literal_eval)
else:
    df['skills'] = [[] for _ in range(len(df))]

if 'professional_company_names' in df.columns:
    df['professional_company_names'] = df['professional_company_names'].apply(safe_literal_eval)
else:
    df['professional_company_names'] = [[] for _ in range(len(df))]

if 'positions' in df.columns:
    df['positions'] = df['positions'].apply(safe_literal_eval)
else:
    df['positions'] = [[] for _ in range(len(df))]

# Create combined resume text from multiple columns
def create_resume_text(row):
    text_parts = []
    
    # Career objective
    if 'career_objective' in df.columns and pd.notna(row['career_objective']):
        text_parts.append(safe_str(row['career_objective']))
    
    # Skills
    if row['skills'] and len(row['skills']) > 0:
        skill_str = ' '.join([str(s) for s in row['skills'] if s])
        if skill_str:
            text_parts.append(skill_str)
    
    # Responsibilities
    if responsibilities_col and pd.notna(row[responsibilities_col]):
        text_parts.append(safe_str(row[responsibilities_col]))
    
    # Positions held
    if row['positions'] and len(row['positions']) > 0:
        pos_str = ' '.join([str(p) for p in row['positions'] if p])
        if pos_str:
            text_parts.append(pos_str)
    
    # Companies
    if row['professional_company_names'] and len(row['professional_company_names']) > 0:
        comp_str = ' '.join([str(c) for c in row['professional_company_names'] if c])
        if comp_str:
            text_parts.append(comp_str)
    
    # Education
    if 'degree_names' in df.columns and pd.notna(row['degree_names']):
        text_parts.append(safe_str(row['degree_names']))
    
    if 'major_field_of_studies' in df.columns and pd.notna(row['major_field_of_studies']):
        text_parts.append(safe_str(row['major_field_of_studies']))
    
    # Educational institution
    if 'educational_institution_name' in df.columns and pd.notna(row['educational_institution_name']):
        text_parts.append(safe_str(row['educational_institution_name']))
    
    result = ' '.join(text_parts).lower()
    return result if result and len(result) > 10 else "no content available"

df['resume_text'] = df.apply(create_resume_text, axis=1)

# Create job requirements text
def create_job_text(row):
    text_parts = []
    
    if job_position_col and pd.notna(row[job_position_col]):
        text_parts.append(safe_str(row[job_position_col]))
    
    if skills_req_col and pd.notna(row[skills_req_col]):
        text_parts.append(safe_str(row[skills_req_col]))
    
    if edu_req_col and pd.notna(row[edu_req_col]):
        text_parts.append(safe_str(row[edu_req_col]))
    
    if exp_req_col and pd.notna(row[exp_req_col]):
        text_parts.append(safe_str(row[exp_req_col]))
    
    if responsibilities_col and pd.notna(row[responsibilities_col]):
        text_parts.append(safe_str(row[responsibilities_col]))
    
    result = ' '.join(text_parts).lower()
    return result if result and len(result) > 5 else "no job description available"

df['job_text'] = df.apply(create_job_text, axis=1)

# Clean matched_score
if 'matched_score' in df.columns:
    df['matched_score'] = pd.to_numeric(df['matched_score'], errors='coerce')
else:
    # Check for similar column names
    score_cols = [col for col in df.columns if 'score' in col.lower()]
    if score_cols:
        print(f"\nUsing '{score_cols[0]}' as score column")
        df['matched_score'] = pd.to_numeric(df[score_cols[0]], errors='coerce')
    else:
        print("\nNo score column found. Creating synthetic scores for demo.")
        df['matched_score'] = np.random.uniform(0.3, 0.9, len(df))

# Remove rows with missing scores or empty texts
df_clean = df.dropna(subset=['matched_score']).copy()
df_clean = df_clean[df_clean['resume_text'] != "no content available"]
df_clean = df_clean[df_clean['job_text'] != "no job description available"]

print(f"\nClean dataset shape: {df_clean.shape}")

if len(df_clean) == 0:
    print("\n❌ ERROR: No valid data found after cleaning!")
    print("Please check your CSV file structure.")
    exit()

print(f"Score range: {df_clean['matched_score'].min():.3f} to {df_clean['matched_score'].max():.3f}")

# Show sample
print("\n" + "="*50)
print("SAMPLE DATA")
print("="*50)
print("\nSample resume text (first 300 chars):")
print(df_clean['resume_text'].iloc[0][:300])
print("\nSample job text (first 300 chars):")
print(df_clean['job_text'].iloc[0][:300])

# ============================================
# STEP 4: CREATE FEATURES
# ============================================

print("\n" + "="*50)
print("CREATING FEATURES")
print("="*50)

# Feature 1: TF-IDF similarity between resume and job
tfidf = TfidfVectorizer(max_features=500, stop_words='english', max_df=0.8, min_df=2)

# Combine resume and job texts
all_texts = df_clean['resume_text'].tolist() + df_clean['job_text'].tolist()

try:
    tfidf.fit(all_texts)
    resume_vectors = tfidf.transform(df_clean['resume_text'])
    job_vectors = tfidf.transform(df_clean['job_text'])
    
    # Cosine similarity
    df_clean['similarity_score'] = [
        cosine_similarity(resume_vectors[i], job_vectors[i])[0][0] 
        for i in range(len(df_clean))
    ]
    print(f"✓ Similarity scores calculated (range: {df_clean['similarity_score'].min():.3f} to {df_clean['similarity_score'].max():.3f})")
except Exception as e:
    print(f"Error in similarity calculation: {e}")
    df_clean['similarity_score'] = 0.5

# Feature 2: Extract years of experience
def extract_experience_years(text):
    if not text or text == "no content available":
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

df_clean['exp_years'] = df_clean['resume_text'].apply(extract_experience_years)
print(f"✓ Experience extracted (max: {df_clean['exp_years'].max()} years)")

# Feature 3: Count of skills
df_clean['skill_count'] = df_clean['skills'].apply(len)
print(f"✓ Skill count calculated (max: {df_clean['skill_count'].max()} skills)")

# Feature 4: Education level
def get_education_level(degree):
    if pd.isna(degree) or degree is None:
        return 0
    degree_str = str(degree).lower()
    if 'phd' in degree_str or 'doctor' in degree_str:
        return 5
    elif 'master' in degree_str or 'm.tech' in degree_str or 'msc' in degree_str:
        return 4
    elif 'bachelor' in degree_str or 'b.tech' in degree_str or 'b.e' in degree_str or 'b.sc' in degree_str:
        return 3
    elif 'diploma' in degree_str:
        return 2
    elif 'high school' in degree_str or '12th' in degree_str:
        return 1
    return 0

if 'degree_names' in df_clean.columns:
    df_clean['edu_level'] = df_clean['degree_names'].apply(get_education_level)
else:
    df_clean['edu_level'] = 0
print(f"✓ Education level calculated")

# Feature 5: Resume length
df_clean['resume_length'] = df_clean['resume_text'].apply(lambda x: len(x.split()))
print(f"✓ Resume length calculated (avg: {df_clean['resume_length'].mean():.0f} words)")

# ============================================
# STEP 5: TRAIN MODEL
# ============================================

print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

feature_columns = ['similarity_score', 'exp_years', 'skill_count', 'edu_level', 'resume_length']
X = df_clean[feature_columns].fillna(0)
y = df_clean['matched_score'].fillna(df_clean['matched_score'].mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  R² Score: {r2:.4f}")
print(f"\nFeature Importance:")
for col, imp in zip(feature_columns, model.feature_importances_):
    print(f"  {col}: {imp:.4f}")

# ============================================
# STEP 6: ANALYSIS FUNCTION
# ============================================

def analyze_resume(resume_text, job_description):
    try:
        resume_vec = tfidf.transform([resume_text.lower()])
        job_vec = tfidf.transform([job_description.lower()])
        sim_score = cosine_similarity(resume_vec, job_vec)[0][0]
        exp_years = extract_experience_years(resume_text)
        resume_len = len(resume_text.split())
        
        features = np.array([[sim_score, exp_years, 0, 0, resume_len]])
        predicted_score = model.predict(features)[0]
        
        return {
            'predicted_match_score': round(predicted_score, 3),
            'similarity_score': round(sim_score, 3),
            'extracted_experience': exp_years,
            'confidence': 'High' if predicted_score > 0.7 else 'Medium' if predicted_score > 0.4 else 'Low'
        }
    except Exception as e:
        return {'error': str(e), 'predicted_match_score': 0.5}

# ============================================
# STEP 7: EXAMPLE
# ============================================

print("\n" + "="*50)
print("EXAMPLE PREDICTION")
print("="*50)

sample_resume = df_clean['resume_text'].iloc[0]
sample_job = df_clean['job_text'].iloc[0]
actual_score = df_clean['matched_score'].iloc[0]

result = analyze_resume(sample_resume, sample_job)

print(f"Actual match score: {actual_score:.3f}")
print(f"Predicted match score: {result['predicted_match_score']}")
print(f"Similarity score: {result['similarity_score']}")
print(f"Extracted experience: {result['extracted_experience']} years")
print(f"Confidence: {result.get('confidence', 'N/A')}")

# ============================================
# STEP 8: SAVE MODEL
# ============================================

import joblib

try:
    joblib.dump(model, 'resume_analyzer_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("\n" + "="*50)
    print("✅ MODEL SAVED SUCCESSFULLY")
    print("="*50)
    print("Files created:")
    print("  - resume_analyzer_model.pkl")
    print("  - tfidf_vectorizer.pkl")
except Exception as e:
    print(f"\nWarning: Could not save model: {e}")

print("\n✅ Done!")