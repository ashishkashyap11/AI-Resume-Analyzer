"""
Job Matcher Module
Matches resumes against job descriptions using a HYBRID approach:
1. Semantic Similarity (Sentence-Transformers) - Meaning & context
2. TF-IDF Similarity - Keyword & terminology match
3. Skill Gap Analysis - Direct required/preferred skill comparison
4. Configurable Hybrid Scoring - Balanced multi-metric score

Author: Ashish Kashyap
Project: AI Resume Analyzer - EduVitae Services
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import semantic matcher
try:
    from utils.semantic_matcher import calculate_semantic_similarity, get_model_info
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: Semantic matcher not available, falling back to TF-IDF only.")


# ──────────────────────────────────────────────────────────────
# Configurable Scoring Weights
# ──────────────────────────────────────────────────────────────

# Default hybrid scoring weights (must sum to 1.0)
DEFAULT_SCORING_WEIGHTS = {
    'semantic': 0.35,      # Meaning and context match
    'tfidf': 0.30,         # Exact keyword and terminology match
    'required': 0.25,      # Essential required skills
    'preferred': 0.10,     # Bonus/preferred skills
}

# Fallback weights when semantic matching is unavailable
FALLBACK_SCORING_WEIGHTS = {
    'tfidf': 0.40,
    'required': 0.45,
    'preferred': 0.15,
}


# ──────────────────────────────────────────────────────────────
# Sample Job Descriptions Database
# ──────────────────────────────────────────────────────────────

JOB_DESCRIPTIONS = {
    'data_scientist': {
        'title': 'Data Scientist',
        'company': 'Tech Corp',
        'description': """
        We are looking for a Data Scientist with strong skills in Python,
        Machine Learning, Deep Learning, and Statistical Analysis. The ideal
        candidate should have experience with TensorFlow or PyTorch, data
        visualization using Matplotlib and Seaborn, and proficiency in SQL
        for database management. Experience with NLP, Computer Vision, and
        cloud platforms (AWS/GCP) is a plus. Must have strong problem-solving
        and communication skills.
        """,
        'required_skills': [
            'python', 'machine learning', 'deep learning', 'sql',
            'tensorflow', 'pandas', 'numpy', 'matplotlib', 'statistics',
            'data visualization', 'scikit-learn',
        ],
        'preferred_skills': [
            'pytorch', 'nlp', 'computer vision', 'aws', 'gcp',
            'spark', 'docker', 'git', 'tableau', 'power bi',
        ],
    },
    'frontend_developer': {
        'title': 'Frontend Developer',
        'company': 'Web Solutions Inc',
        'description': """
        Looking for a Frontend Developer proficient in React, JavaScript,
        TypeScript, HTML, CSS, and modern web technologies. Experience with
        Next.js, Redux, REST APIs, and responsive design is required.
        Knowledge of testing frameworks like Jest and Cypress is preferred.
        Must be familiar with Git, Agile methodologies, and have strong
        UI/UX sensibility.
        """,
        'required_skills': [
            'react', 'javascript', 'html', 'css', 'typescript',
            'git', 'rest api', 'responsive design',
        ],
        'preferred_skills': [
            'nextjs', 'redux', 'jest', 'cypress', 'tailwind',
            'sass', 'webpack', 'figma', 'agile', 'graphql',
        ],
    },
    'backend_developer': {
        'title': 'Backend Developer',
        'company': 'Server Systems Ltd',
        'description': """
        Seeking a Backend Developer with expertise in Python/Node.js,
        Django/Flask, REST API development, and database management with
        PostgreSQL and MongoDB. Must have experience with Docker, CI/CD
        pipelines, and cloud services. Knowledge of microservices
        architecture, Redis, and message queues is a plus.
        """,
        'required_skills': [
            'python', 'django', 'flask', 'rest api', 'postgresql',
            'sql', 'git', 'docker',
        ],
        'preferred_skills': [
            'nodejs', 'mongodb', 'redis', 'kubernetes', 'aws',
            'ci/cd', 'microservices', 'nginx', 'linux', 'kafka',
        ],
    },
    'ml_engineer': {
        'title': 'Machine Learning Engineer',
        'company': 'AI Innovations',
        'description': """
        We need an ML Engineer with strong Python skills, experience in
        building and deploying machine learning models. Proficiency in
        TensorFlow, PyTorch, Scikit-learn, and MLOps tools is required.
        Experience with Docker, Kubernetes, cloud platforms, and building
        data pipelines. Knowledge of LLMs, RAG, and NLP is highly valued.
        """,
        'required_skills': [
            'python', 'machine learning', 'tensorflow', 'scikit-learn',
            'docker', 'sql', 'pandas', 'numpy', 'git',
        ],
        'preferred_skills': [
            'pytorch', 'kubernetes', 'mlops', 'aws', 'spark',
            'llm', 'rag', 'nlp', 'fastapi', 'airflow',
        ],
    },
    'data_analyst': {
        'title': 'Data Analyst',
        'company': 'Analytics Pro',
        'description': """
        Looking for a Data Analyst skilled in SQL, Python, Excel, and data
        visualization tools like Tableau or Power BI. Must be able to
        perform exploratory data analysis, create reports, build dashboards,
        and communicate insights to stakeholders. Experience with statistical
        analysis, A/B testing, and Pandas is required.
        """,
        'required_skills': [
            'sql', 'python', 'excel', 'data visualization',
            'pandas', 'data analysis', 'statistics',
        ],
        'preferred_skills': [
            'tableau', 'power bi', 'matplotlib', 'seaborn',
            'numpy', 'a/b testing', 'git', 'r', 'jupyter',
        ],
    },
    'fullstack_developer': {
        'title': 'Full Stack Developer',
        'company': 'Digital Dynamics',
        'description': """
        Hiring a Full Stack Developer with expertise in both frontend
        (React/Angular, HTML, CSS, JavaScript) and backend (Node.js/Python,
        Express/Django). Must have experience with databases (MongoDB,
        PostgreSQL), REST APIs, Git, and deployment. Knowledge of Docker,
        AWS, and CI/CD pipelines is preferred.
        """,
        'required_skills': [
            'javascript', 'react', 'nodejs', 'html', 'css',
            'mongodb', 'sql', 'rest api', 'git',
        ],
        'preferred_skills': [
            'python', 'django', 'express', 'postgresql', 'docker',
            'aws', 'typescript', 'redis', 'ci/cd', 'angular',
        ],
    },
    'devops_engineer': {
        'title': 'DevOps Engineer',
        'company': 'Cloud Masters',
        'description': """
        We are hiring a DevOps Engineer with expertise in CI/CD, Docker,
        Kubernetes, and cloud platforms (AWS/Azure/GCP). Must have strong
        Linux skills, experience with Terraform, Ansible, monitoring tools,
        and scripting (Python/Bash). Knowledge of microservices and
        networking is essential.
        """,
        'required_skills': [
            'docker', 'kubernetes', 'linux', 'aws', 'ci/cd',
            'git', 'python', 'bash',
        ],
        'preferred_skills': [
            'terraform', 'ansible', 'azure', 'gcp', 'jenkins',
            'nginx', 'monitoring', 'networking', 'microservices', 'helm',
        ],
    },
    'software_engineer': {
        'title': 'Software Engineer',
        'company': 'CodeCraft Solutions',
        'description': """
        Looking for a Software Engineer with strong programming skills in
        Python, Java, or C++. Must have solid understanding of data
        structures, algorithms, OOP, and system design. Experience with
        databases, version control (Git), and testing. Knowledge of web
        frameworks, cloud, and agile development practices is a plus.
        """,
        'required_skills': [
            'python', 'java', 'c++', 'data structures', 'algorithms',
            'sql', 'git', 'oop',
        ],
        'preferred_skills': [
            'system design', 'docker', 'aws', 'rest api', 'agile',
            'testing', 'linux', 'django', 'spring', 'redis',
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# TF-IDF Matching (Existing Implementation Preserved)
# ──────────────────────────────────────────────────────────────

def calculate_tfidf_similarity(resume_text, job_description):
    """
    Calculate similarity between resume and job description
    using TF-IDF vectorization and cosine similarity.

    Preserved from original implementation for backward compatibility
    and keyword-level matching.
    """
    if not resume_text or not job_description:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
        )

        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        return round(float(similarity) * 100, 2)
    except Exception as e:
        print(f"Error calculating TF-IDF similarity: {e}")
        return 0.0


# ──────────────────────────────────────────────────────────────
# Skill Matching & Gap Analysis
# ──────────────────────────────────────────────────────────────

def analyze_skill_match(resume_skills, job_required, job_preferred):
    """
    Analyze skill match between resume and job requirements.
    Returns matched, missing required, and missing preferred skills.
    """
    resume_skills_lower = {s.lower() for s in resume_skills}

    matched_required = []
    missing_required = []
    matched_preferred = []
    missing_preferred = []

    for skill in job_required:
        if skill.lower() in resume_skills_lower:
            matched_required.append(skill.title())
        else:
            missing_required.append(skill.title())

    for skill in job_preferred:
        if skill.lower() in resume_skills_lower:
            matched_preferred.append(skill.title())
        else:
            missing_preferred.append(skill.title())

    total_required = len(job_required)
    matched_required_count = len(matched_required)

    required_match_pct = (matched_required_count / total_required * 100) if total_required > 0 else 0

    return {
        'matched_required': matched_required,
        'missing_required': missing_required,
        'matched_preferred': matched_preferred,
        'missing_preferred': missing_preferred,
        'required_match_percentage': round(required_match_pct, 1),
    }


# ──────────────────────────────────────────────────────────────
# Hybrid Scoring System
# ──────────────────────────────────────────────────────────────

def calculate_hybrid_score(semantic_score, tfidf_score, skill_analysis, weights=None):
    """
    Calculate hybrid match score combining:
    1. Semantic similarity (Sentence-Transformers)
    2. TF-IDF similarity (Keyword overlap)
    3. Required skills match percentage
    4. Preferred skills match percentage

    Args:
        semantic_score (float): Semantic similarity (0-100)
        tfidf_score (float): TF-IDF similarity (0-100)
        skill_analysis (dict): Output from analyze_skill_match()
        weights (dict, optional): Custom weights dictionary

    Returns:
        float: Hybrid score (0-100)
    """
    if weights is None:
        weights = DEFAULT_SCORING_WEIGHTS

    required_pct = skill_analysis['required_match_percentage']

    total_preferred = len(skill_analysis['matched_preferred']) + len(skill_analysis['missing_preferred'])
    preferred_pct = (len(skill_analysis['matched_preferred']) / total_preferred * 100) if total_preferred > 0 else 0

    # Calculate weighted hybrid score
    hybrid = (
        semantic_score * weights.get('semantic', 0.35) +
        tfidf_score * weights.get('tfidf', 0.30) +
        required_pct * weights.get('required', 0.25) +
        preferred_pct * weights.get('preferred', 0.10)
    )

    return round(min(100.0, max(0.0, hybrid)), 1)


def calculate_overall_score(tfidf_score, skill_analysis):
    """
    Legacy overall score calculation (preserved for backward compatibility).
    Uses the original 40/45/15 weighting formula.
    """
    tfidf_weight = 0.40
    required_weight = 0.45
    preferred_weight = 0.15

    required_pct = skill_analysis['required_match_percentage']

    total_preferred = len(skill_analysis['matched_preferred']) + len(skill_analysis['missing_preferred'])
    preferred_pct = (len(skill_analysis['matched_preferred']) / total_preferred * 100) if total_preferred > 0 else 0

    overall = (tfidf_score * tfidf_weight +
               required_pct * required_weight +
               preferred_pct * preferred_weight)

    return round(min(overall, 100), 1)


def get_score_label(score):
    """Return a descriptive label for the match score."""
    if score >= 80:
        return {'label': 'Excellent Match', 'color': 'success', 'emoji': '🟢'}
    elif score >= 60:
        return {'label': 'Good Match', 'color': 'primary', 'emoji': '🔵'}
    elif score >= 40:
        return {'label': 'Moderate Match', 'color': 'warning', 'emoji': '🟡'}
    else:
        return {'label': 'Low Match', 'color': 'danger', 'emoji': '🔴'}


# ──────────────────────────────────────────────────────────────
# Recommendations Engine
# ──────────────────────────────────────────────────────────────

def generate_recommendations(skill_analysis, score):
    """Generate actionable recommendations based on skill gap analysis."""
    recommendations = []

    missing_req = skill_analysis['missing_required']
    missing_pref = skill_analysis['missing_preferred']

    if missing_req:
        recommendations.append({
            'type': 'critical',
            'icon': '🔴',
            'title': 'Missing Required Skills',
            'detail': f"Learn these skills to significantly improve your match: {', '.join(missing_req[:5])}",
        })

    if missing_pref:
        recommendations.append({
            'type': 'improvement',
            'icon': '🟡',
            'title': 'Missing Preferred Skills',
            'detail': f"These skills would give you an edge: {', '.join(missing_pref[:5])}",
        })

    if skill_analysis['required_match_percentage'] >= 80:
        recommendations.append({
            'type': 'positive',
            'icon': '🟢',
            'title': 'Strong Skill Match',
            'detail': 'Your required skills match well! Focus on highlighting relevant projects.',
        })

    if score < 50:
        recommendations.append({
            'type': 'improvement',
            'icon': '💡',
            'title': 'Improve Resume Keywords',
            'detail': 'Add more job-relevant keywords and quantify achievements with numbers.',
        })

    recommendations.append({
        'type': 'tip',
        'icon': '📝',
        'title': 'Resume Tip',
        'detail': 'Tailor your resume for each job application. Use keywords from the job description.',
    })

    return recommendations


# ──────────────────────────────────────────────────────────────
# Main Matching Functions (Updated with Hybrid Approach)
# ──────────────────────────────────────────────────────────────

def match_resume_to_job(resume_text, resume_skills, job_key, weights=None):
    """
    Main matching function: Match a resume against a specific job.
    Returns comprehensive analysis with BOTH separate and hybrid scores.

    Backward compatible: Maintains all existing return fields while adding new ones.
    """
    if job_key not in JOB_DESCRIPTIONS:
        return {'error': f'Job role "{job_key}" not found.'}

    job = JOB_DESCRIPTIONS[job_key]

    # 1. Calculate TF-IDF similarity (keyword matching)
    tfidf_score = calculate_tfidf_similarity(resume_text, job['description'])

    # 2. Calculate Semantic similarity (meaning matching)
    if SEMANTIC_AVAILABLE:
        semantic_score = calculate_semantic_similarity(resume_text, job['description'])
    else:
        semantic_score = tfidf_score  # Fallback

    # 3. Analyze skill match
    all_resume_skills = resume_skills.get('technical', []) + resume_skills.get('soft', [])
    skill_analysis = analyze_skill_match(
        all_resume_skills,
        job['required_skills'],
        job['preferred_skills']
    )

    # 4. Calculate scores
    if SEMANTIC_AVAILABLE:
        hybrid_score = calculate_hybrid_score(semantic_score, tfidf_score, skill_analysis, weights)
        overall_score = hybrid_score  # Use hybrid score as primary
    else:
        hybrid_score = calculate_overall_score(tfidf_score, skill_analysis)
        overall_score = hybrid_score

    score_label = get_score_label(overall_score)
    recommendations = generate_recommendations(skill_analysis, overall_score)

    return {
        # Core details
        'job_title': job['title'],
        'job_company': job['company'],

        # Separate metrics
        'semantic_score': semantic_score,      # NEW: Semantic similarity
        'tfidf_score': tfidf_score,            # EXISTING: TF-IDF similarity
        'hybrid_score': hybrid_score,          # NEW: Hybrid score
        'overall_score': overall_score,        # EXISTING: (backward compat)

        # Skill & gap analysis
        'skill_analysis': skill_analysis,
        'score_label': score_label,
        'recommendations': recommendations,

        # Metadata
        'is_hybrid': SEMANTIC_AVAILABLE,
        'model_used': 'all-MiniLM-L6-v2' if SEMANTIC_AVAILABLE else 'TF-IDF only',
        'scoring_weights': weights or (DEFAULT_SCORING_WEIGHTS if SEMANTIC_AVAILABLE else FALLBACK_SCORING_WEIGHTS),
    }


def match_all_jobs(resume_text, resume_skills, weights=None):
    """
    Match resume against ALL job descriptions and rank them.
    Uses hybrid scoring for ranking.
    """
    results = []

    for key in JOB_DESCRIPTIONS:
        result = match_resume_to_job(resume_text, resume_skills, key, weights)
        if 'error' not in result:
            result['job_key'] = key
            results.append(result)

    # Sort by overall/hybrid score descending
    results.sort(key=lambda x: x['overall_score'], reverse=True)

    return results


def get_job_list():
    """Return list of available job roles."""
    return [
        {'key': key, 'title': job['title'], 'company': job['company']}
        for key, job in JOB_DESCRIPTIONS.items()
    ]
