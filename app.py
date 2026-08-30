"""
AI Resume Analyzer - Flask Web Application
Main application entry point.

Features:
- Upload resume (PDF/DOCX)
- Extract skills, education, experience using NLP
- Match resume against job descriptions using TF-IDF & Cosine Similarity
- Skill gap analysis with actionable recommendations
- Interactive web dashboard with scoring

Author: Ashish Kashyap
Project: AI Resume Analyzer - EduVitae Services
Tech: Python, Flask, Scikit-learn, NLP, TF-IDF, Cosine Similarity
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from utils.resume_parser import parse_resume
from utils.job_matcher import match_resume_to_job, match_all_jobs, get_job_list

# ──────────────────────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Landing page with upload form."""
    jobs = get_job_list()
    return render_template('index.html', jobs=jobs)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle resume upload and analysis."""
    # Validate file upload
    if 'resume' not in request.files:
        flash('No file uploaded. Please select a resume file.', 'error')
        return redirect(url_for('index'))

    file = request.files['resume']

    if file.filename == '':
        flash('No file selected. Please choose a PDF or DOCX file.', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file format. Only PDF and DOCX files are supported.', 'error')
        return redirect(url_for('index'))

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Parse the resume
        parsed = parse_resume(filepath)

        if 'error' in parsed:
            flash(parsed['error'], 'error')
            return redirect(url_for('index'))

        # Get selected job role
        job_role = request.form.get('job_role', '')

        # Match against jobs
        if job_role and job_role != 'all':
            job_match = match_resume_to_job(
                parsed['raw_text'],
                parsed['skills'],
                job_role
            )
            all_matches = None
        else:
            job_match = None
            all_matches = match_all_jobs(
                parsed['raw_text'],
                parsed['skills']
            )

        return render_template(
            'results.html',
            parsed=parsed,
            job_match=job_match,
            all_matches=all_matches,
            selected_role=job_role,
            jobs=get_job_list(),
        )

    except Exception as e:
        flash(f'Error analyzing resume: {str(e)}', 'error')
        return redirect(url_for('index'))

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/jobs', methods=['GET'])
def api_jobs():
    """API endpoint to get available job roles."""
    return jsonify(get_job_list())


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
