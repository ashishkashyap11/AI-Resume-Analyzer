"""
Resume Parser Module
Extracts key information from resumes including:
- Contact Information (Name, Email, Phone, LinkedIn)
- Skills (Technical & Soft)
- Education
- Experience
- Certifications

Author: Ashish Kashyap
Project: AI Resume Analyzer - EduVitae Services
"""

import re
import os
import PyPDF2
import docx
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ──────────────────────────────────────────────────────────────
# Skill Database — categorized technical and soft skills
# ──────────────────────────────────────────────────────────────

TECHNICAL_SKILLS = {
    # Programming Languages
    'python', 'java', 'javascript', 'c++', 'c#', 'c', 'ruby', 'go', 'golang',
    'rust', 'swift', 'kotlin', 'typescript', 'scala', 'r', 'matlab', 'perl',
    'php', 'dart', 'lua', 'haskell', 'elixir', 'clojure', 'sql', 'nosql',
    'bash', 'shell', 'powershell',

    # Web Frameworks
    'react', 'angular', 'vue', 'vuejs', 'nextjs', 'next.js', 'nuxt',
    'django', 'flask', 'fastapi', 'express', 'expressjs', 'spring boot',
    'spring', 'rails', 'ruby on rails', 'laravel', 'asp.net', 'svelte',
    'gatsby', 'remix', 'nestjs',

    # Data Science & ML
    'machine learning', 'deep learning', 'neural networks', 'tensorflow',
    'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
    'matplotlib', 'seaborn', 'plotly', 'scipy', 'opencv', 'nlp',
    'natural language processing', 'computer vision', 'transformer',
    'bert', 'gpt', 'llm', 'rag', 'langchain', 'huggingface',
    'xgboost', 'lightgbm', 'random forest', 'svm', 'regression',
    'classification', 'clustering', 'reinforcement learning',
    'data analysis', 'data visualization', 'data engineering',
    'feature engineering', 'eda', 'exploratory data analysis',
    'statistical modeling', 'a/b testing', 'hypothesis testing',
    'spacy', 'nltk', 'text mining', 'sentiment analysis',

    # Databases
    'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
    'dynamodb', 'cassandra', 'elasticsearch', 'firebase', 'supabase',
    'neo4j', 'mariadb', 'couchdb', 'influxdb',

    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
    'jenkins', 'ci/cd', 'terraform', 'ansible', 'nginx', 'apache',
    'heroku', 'vercel', 'netlify', 'digitalocean', 'linux', 'unix',
    'git', 'github', 'gitlab', 'bitbucket',

    # Tools & Others
    'rest api', 'graphql', 'websocket', 'microservices', 'api',
    'html', 'css', 'sass', 'tailwind', 'bootstrap', 'nodejs',
    'node.js', 'webpack', 'vite', 'babel', 'jest', 'pytest',
    'selenium', 'cypress', 'postman', 'jira', 'confluence',
    'figma', 'adobe', 'tableau', 'power bi', 'excel', 'ms excel',
    'vs code', 'jupyter', 'colab', 'streamlit', 'gradio',
    'agile', 'scrum', 'kanban', 'devops', 'mlops',
    'hadoop', 'spark', 'kafka', 'airflow', 'dbt',
    'blockchain', 'solidity', 'web3',
}

SOFT_SKILLS = {
    'leadership', 'communication', 'teamwork', 'problem solving',
    'critical thinking', 'time management', 'adaptability', 'creativity',
    'collaboration', 'analytical', 'decision making', 'project management',
    'presentation', 'negotiation', 'conflict resolution', 'mentoring',
    'strategic thinking', 'attention to detail', 'multitasking',
    'self motivated', 'work ethic', 'interpersonal',
}

# ──────────────────────────────────────────────────────────────
# Education keywords for detection
# ──────────────────────────────────────────────────────────────

EDUCATION_KEYWORDS = [
    'b.tech', 'btech', 'b.e', 'bachelor', 'master', 'm.tech', 'mtech',
    'm.s', 'ms', 'mba', 'phd', 'doctorate', 'diploma', 'bsc', 'msc',
    'b.sc', 'm.sc', 'bca', 'mca', 'b.com', 'm.com', 'engineering',
    'university', 'college', 'institute', 'school', 'cgpa', 'gpa',
    'aggregate', 'percentage', 'class 10', 'class 12', '10th', '12th',
    'ssc', 'hsc', 'cbse', 'icse',
]

DEGREE_PATTERNS = [
    r'(?i)b\.?tech|bachelor\s+of\s+technology',
    r'(?i)m\.?tech|master\s+of\s+technology',
    r'(?i)b\.?e\.?|bachelor\s+of\s+engineering',
    r'(?i)m\.?s\.?|master\s+of\s+science',
    r'(?i)mba|master\s+of\s+business',
    r'(?i)b\.?sc|bachelor\s+of\s+science',
    r'(?i)m\.?sc|master\s+of\s+science',
    r'(?i)ph\.?d|doctorate',
    r'(?i)bca|mca',
    r'(?i)diploma',
]


def extract_text_from_pdf(file_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text


def extract_text_from_docx(file_path):
    """Extract text content from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return text


def extract_text(file_path):
    """Extract text from PDF or DOCX files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def extract_contact_info(text):
    """Extract contact information from resume text."""
    contact = {
        'name': '',
        'email': '',
        'phone': '',
        'linkedin': '',
        'github': '',
    }

    # Extract email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    if emails:
        contact['email'] = emails[0]

    # Extract phone number (Indian & international formats)
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
    phones = re.findall(phone_pattern, text)
    if phones:
        contact['phone'] = phones[0].strip()

    # Extract LinkedIn URL
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+'
    linkedin = re.findall(linkedin_pattern, text)
    if linkedin:
        contact['linkedin'] = linkedin[0]

    # Extract GitHub URL
    github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+'
    github = re.findall(github_pattern, text)
    if github:
        contact['github'] = github[0]

    # Extract name (usually the first line of the resume)
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 2 and len(line) < 50:
            # Check if it looks like a name (no special chars, no email/phone)
            if not re.search(r'[@\d]', line) and not any(kw in line.lower() for kw in ['resume', 'cv', 'curriculum']):
                contact['name'] = line
                break

    return contact


def extract_skills(text):
    """Extract technical and soft skills from resume text."""
    text_lower = text.lower()
    found_technical = []
    found_soft = []

    # Check for technical skills
    for skill in TECHNICAL_SKILLS:
        # Use word boundary matching for short skills to avoid false positives
        if len(skill) <= 2:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_technical.append(skill.title())
        else:
            if skill in text_lower:
                found_technical.append(skill.title())

    # Check for soft skills
    for skill in SOFT_SKILLS:
        if skill in text_lower:
            found_soft.append(skill.title())

    return {
        'technical': sorted(list(set(found_technical))),
        'soft': sorted(list(set(found_soft))),
        'total_count': len(set(found_technical)) + len(set(found_soft)),
    }


def extract_education(text):
    """Extract education details from resume text."""
    education = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        # Check if line contains education keywords
        for keyword in EDUCATION_KEYWORDS:
            if keyword in line_lower:
                # Gather context (current line + next 2 lines)
                edu_text = line.strip()
                for j in range(1, 3):
                    if i + j < len(lines) and lines[i + j].strip():
                        edu_text += " | " + lines[i + j].strip()

                # Extract GPA/CGPA if present
                gpa_match = re.search(r'(?:cgpa|gpa|aggregate)[:\s]*(\d+\.?\d*)', line_lower)
                percentage_match = re.search(r'(\d+\.?\d*)\s*%', line_lower)

                edu_entry = {
                    'text': edu_text,
                    'gpa': gpa_match.group(1) if gpa_match else None,
                    'percentage': percentage_match.group(1) if percentage_match else None,
                }

                # Avoid duplicate entries
                if not any(edu_entry['text'] == e['text'] for e in education):
                    education.append(edu_entry)
                break

    return education


def extract_experience(text):
    """Extract work experience sections from resume text."""
    experience = []
    lines = text.split('\n')

    experience_keywords = [
        'experience', 'internship', 'work history', 'employment',
        'professional experience', 'work experience',
    ]

    in_experience_section = False
    current_exp = []

    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Detect experience section start
        if any(kw in line_lower for kw in experience_keywords) and len(line_stripped) < 50:
            in_experience_section = True
            continue

        # Detect section end (new major section)
        if in_experience_section and line_stripped:
            section_headers = ['education', 'skills', 'certifications', 'projects',
                               'awards', 'achievements', 'hobbies', 'references']
            if any(line_lower.startswith(h) or line_lower == h for h in section_headers):
                in_experience_section = False
                if current_exp:
                    experience.append('\n'.join(current_exp))
                    current_exp = []
                continue

        if in_experience_section and line_stripped:
            current_exp.append(line_stripped)

    if current_exp:
        experience.append('\n'.join(current_exp))

    return experience


def extract_certifications(text):
    """Extract certifications from resume text."""
    certifications = []
    lines = text.split('\n')

    in_cert_section = False

    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        if 'certification' in line_lower and len(line_stripped) < 30:
            in_cert_section = True
            continue

        if in_cert_section:
            section_headers = ['education', 'skills', 'experience', 'projects',
                               'awards', 'technical', 'hobbies']
            if any(line_lower.startswith(h) or line_lower == h for h in section_headers):
                in_cert_section = False
                continue

            if line_stripped and len(line_stripped) > 5:
                # Clean up bullet points
                clean = re.sub(r'^[▪•\-\*]\s*', '', line_stripped)
                if clean:
                    certifications.append(clean)

    return certifications


def parse_resume(file_path):
    """
    Main function: Parse a resume file and extract all information.
    Returns a structured dictionary with all extracted data.
    """
    text = extract_text(file_path)

    if not text.strip():
        return {'error': 'Could not extract text from the file.'}

    result = {
        'raw_text': text,
        'contact': extract_contact_info(text),
        'skills': extract_skills(text),
        'education': extract_education(text),
        'experience': extract_experience(text),
        'certifications': extract_certifications(text),
        'word_count': len(text.split()),
        'sections_found': [],
    }

    # Determine which sections were found
    if result['contact']['email'] or result['contact']['phone']:
        result['sections_found'].append('Contact Information')
    if result['skills']['technical']:
        result['sections_found'].append('Technical Skills')
    if result['skills']['soft']:
        result['sections_found'].append('Soft Skills')
    if result['education']:
        result['sections_found'].append('Education')
    if result['experience']:
        result['sections_found'].append('Experience')
    if result['certifications']:
        result['sections_found'].append('Certifications')

    return result
