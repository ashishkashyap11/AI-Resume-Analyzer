# 🤖 AI Resume Analyzer

> Intelligent resume analysis tool using NLP, TF-IDF vectorization, and machine learning to match resumes with job descriptions, identify skill gaps, and provide actionable recommendations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [About](#about)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [License](#license)

---

## 🎯 About

**AI Resume Analyzer** is a web-based tool developed during my internship at **EduVitae Services** (June 2025 - July 2025). It helps job seekers understand how well their resume aligns with target job roles by:

- **Parsing resumes** using Natural Language Processing (NLP)
- **Extracting** skills, education, experience, and certifications
- **Matching** resumes against job descriptions using **TF-IDF** and **Cosine Similarity**
- **Identifying skill gaps** (missing required/preferred skills)
- **Providing actionable recommendations** to improve job match scores

This project was built to solve a real-world problem: helping candidates optimize their resumes for Applicant Tracking Systems (ATS) and improve their chances in competitive job markets.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Resume Parsing** | Extracts text from PDF and DOCX files using PyPDF2 and python-docx |
| 🧠 **NLP-Powered Extraction** | Automatically detects skills, education, experience, and contact info using NLTK |
| 🎯 **Job Matching** | Matches resumes against 8 predefined job roles (Data Scientist, ML Engineer, Frontend/Backend Developer, etc.) |
| 📊 **TF-IDF Similarity** | Uses TF-IDF vectorization and cosine similarity to score resume-job fit (0-100%) |
| 🔍 **Skill Gap Analysis** | Identifies missing required and preferred skills with visual breakdown |
| 💡 **Actionable Recommendations** | Provides targeted suggestions to improve match scores |
| 📈 **Interactive Dashboard** | Clean, responsive UI built with Bootstrap 5 |
| ⚡ **Fast & Lightweight** | No external APIs required — runs entirely locally |

---

## 🛠️ Technology Stack

### **Backend**
- **Python 3.8+** — Core programming language
- **Flask 3.0.0** — Web framework
- **Scikit-learn 1.3.2** — TF-IDF vectorization, cosine similarity
- **NLTK 3.8.1** — Natural Language Processing
- **PyPDF2 3.0.1** — PDF text extraction
- **python-docx 1.1.0** — DOCX text extraction
- **Pandas & NumPy** — Data manipulation

### **Frontend**
- **HTML5 & CSS3**
- **Bootstrap 5.3.2** — Responsive UI framework
- **JavaScript** — Client-side interactivity
- **Font Awesome 6.5.0** — Icons

---

## 🔬 How It Works

### **1. Resume Parsing**
- Extracts text from PDF/DOCX files
- Uses regex patterns and NLTK to identify:
  - Contact information (email, phone, LinkedIn, GitHub)
  - Technical & soft skills (matched against a curated database of 200+ skills)
  - Education (degrees, universities, CGPA)
  - Experience sections
  - Certifications

### **2. TF-IDF Vectorization**
- Converts resume text and job description into numerical vectors
- Uses `TfidfVectorizer` with unigrams and bigrams
- Captures term frequency and importance across documents

### **3. Cosine Similarity Calculation**
```python
similarity = cosine_similarity(resume_vector, job_vector)
```
- Measures the angle between two vectors (0 = no similarity, 1 = identical)
- Converted to percentage (0-100%)

### **4. Scoring Algorithm**
```python
Overall Score = (TF-IDF Score × 0.40) + (Required Skills Match × 0.45) + (Preferred Skills Match × 0.15)
```

| Weight | Component | Description |
|--------|-----------|-------------|
| 40% | TF-IDF Content Similarity | How well resume keywords match job description |
| 45% | Required Skills Match | Percentage of required skills found in resume |
| 15% | Preferred Skills Match | Percentage of preferred skills found in resume |

### **5. Skill Gap Analysis**
- Compares resume skills against job requirements
- Categorizes skills as:
  - ✅ **Matched Required** (critical for the role)
  - ❌ **Missing Required** (high priority to learn)
  - ✅ **Matched Preferred** (gives competitive edge)
  - 🟡 **Missing Preferred** (nice to have)

### **6. Recommendations Engine**
Generates personalized suggestions based on:
- Missing required skills
- Missing preferred skills
- Overall match score
- Resume optimization tips

---

## 📁 Project Structure

```
AI-Resume-Analyzer/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment variables template
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
│
├── utils/                      # Core logic modules
│   ├── __init__.py
│   ├── resume_parser.py        # NLP-based resume parsing
│   └── job_matcher.py          # TF-IDF matching & scoring
│
├── templates/                  # HTML templates
│   ├── base.html               # Base template with navbar/footer
│   ├── index.html              # Upload page
│   ├── results.html            # Analysis results dashboard
│   └── about.html              # About page
│
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css           # Custom styles
│   └── js/
│       └── main.js             # Client-side JavaScript
│
├── uploads/                    # Temporary upload directory (gitignored)
└── data/                       # Data directory (optional)
```

---

## 🚀 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)
- Git

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/ashishkashyap11/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### **Step 2: Create Virtual Environment**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download NLTK Data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"
```

### **Step 5: Set Up Environment (Optional)**

**Windows:**
```bash
copy .env.example .env
# Edit .env with your preferred configuration
```

**macOS/Linux:**
```bash
cp .env.example .env
# Edit .env with your preferred configuration
```

---

## 💻 Usage

### **Run the Application**
```bash
python app.py
```

The app will start on `http://localhost:5000`

### **Using the Tool**

1. **Upload Resume**
   - Navigate to `http://localhost:5000`
   - Upload a PDF or DOCX resume (max 16MB)

2. **Select Job Role**
   - Choose a specific role (e.g., Data Scientist, ML Engineer)
   - Or select "Match Against All Roles" to see rankings

3. **View Results**
   - Overall match score (0-100%)
   - TF-IDF similarity score
   - Skill gap analysis
   - Personalized recommendations
   - Extracted resume details (skills, education, certifications)

4. **Available Job Roles**
   - Data Scientist
   - Machine Learning Engineer
   - Frontend Developer
   - Backend Developer
   - Full Stack Developer
   - Data Analyst
   - DevOps Engineer
   - Software Engineer

---

## 📸 Screenshots

### Home Page - Upload Interface
![Home Page - Upload your resume and select target job role](https://via.placeholder.com/800x450/4f46e5/ffffff?text=Upload+Interface+-+Add+Screenshot+Here)

*The landing page features a clean upload interface with drag-and-drop support and job role selection.*

### Analysis Results Dashboard
![Results Dashboard - Comprehensive match analysis with scores and recommendations](https://via.placeholder.com/800x450/10b981/ffffff?text=Results+Dashboard+-+Add+Screenshot+Here)

*The results page displays match scores, skill breakdowns, gap analysis, and actionable recommendations.*

> **Note:** Screenshots will be added after deployment. The application features a modern, responsive design with Bootstrap 5.

---

## 🔮 Future Enhancements

- [ ] Add support for more job roles (customizable job descriptions)
- [ ] Integrate with LinkedIn API for automatic profile parsing
- [ ] Add ATS (Applicant Tracking System) compliance checker
- [ ] Implement user authentication and history tracking
- [ ] Export results as PDF report
- [ ] Add spell-check and grammar analysis
- [ ] Support for more file formats (TXT, RTF)
- [ ] Add real-time job market data integration
- [ ] Multi-language resume support
- [ ] Deploy to cloud (Render/Railway/Vercel)

---

## 👨‍💻 Author

**Ashish Kashyap**  
Computer Science Engineering Student | Data Science Enthusiast

- 🔗 LinkedIn: [linkedin.com/in/ashishkashyap18](https://linkedin.com/in/ashishkashyap18)
- 💻 GitHub: [github.com/ashishkashyap11](https://github.com/ashishkashyap11)

**Education:** B.Tech in Computer Science Engineering, Jaypee University of Information Technology (2022-2026)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **EduVitae Services** — for providing the internship opportunity
- **Scikit-learn** — for powerful ML tools
- **NLTK** — for NLP capabilities
- **Flask** — for the lightweight web framework
- **Bootstrap** — for responsive UI components

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

---

*Built with ❤️ by Ashish Kashyap | Powered by Python, Flask, and Machine Learning*
