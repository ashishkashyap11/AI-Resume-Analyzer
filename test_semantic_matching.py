"""
Test Cases for Semantic & Hybrid Matching System

Tests:
1. Strong match (similar content, similar skills)
2. Weak match (different domain)
3. Semantic vs Keyword test (same meaning, different words)
4. Empty text handling
5. PDF/DOCX resume parsing integration
6. Score comparison (TF-IDF vs Semantic vs Hybrid)

Author: Ashish Kashyap
Project: AI Resume Analyzer - EduVitae Services
"""

import sys
import os

# Set UTF-8 encoding for standard output to avoid UnicodeEncodeError on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.semantic_matcher import calculate_semantic_similarity, get_model_info
from utils.job_matcher import (
    calculate_tfidf_similarity,
    calculate_hybrid_score,
    analyze_skill_match,
    match_resume_to_job
)


# ──────────────────────────────────────────────────────────────
# Test Data
# ──────────────────────────────────────────────────────────────

STRONG_RESUME = """
Data Scientist with 3 years of experience in Python, Machine Learning, and Deep Learning.
Expertise in TensorFlow, PyTorch, and building predictive models. Proficient in SQL,
Pandas, NumPy, and data visualization using Matplotlib and Seaborn. Experience with
NLP projects and deploying ML models on AWS. Strong problem-solving and communication skills.
"""

STRONG_JOB_DESC = """
We are looking for a Data Scientist with strong skills in Python, Machine Learning,
Deep Learning, and Statistical Analysis. Experience with TensorFlow or PyTorch, data
visualization, and SQL required. NLP and cloud platforms (AWS/GCP) experience is a plus.
"""

WEAK_RESUME = """
Frontend Developer with 4 years of experience building responsive web applications using
React, JavaScript, HTML, CSS, and TypeScript. Experienced with Next.js, Redux, REST APIs,
and modern UI/UX design. Familiar with Git, Agile methodologies, and testing frameworks
like Jest and Cypress.
"""

SEMANTIC_RESUME = """
I have worked extensively on artificial intelligence and neural network projects using
Python for the past 3 years. My expertise includes building predictive algorithms,
working with large datasets, and creating data-driven visualizations. I'm skilled in
database queries and have deployed models to cloud infrastructure.
"""

MINIMAL_SKILLS_RESUME = """
Software Engineer with basic Python knowledge. Familiar with programming fundamentals
and eager to learn new technologies.
"""


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)


def test_model_info():
    """Test 1: Verify semantic model information."""
    print_separator("TEST 1: Model Information")

    info = get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Size: {info['model_size']}")
    print(f"Dimensions: {info['embedding_dimensions']}")
    print(f"Speed: {info['speed']}")
    print(f"Use Case: {info['use_case']}")
    print("\n✅ Model info retrieved successfully")


def test_strong_match():
    """Test 2: Strong match - similar content and skills."""
    print_separator("TEST 2: Strong Match (Data Scientist Resume vs DS Job)")

    semantic_score = calculate_semantic_similarity(STRONG_RESUME, STRONG_JOB_DESC)
    tfidf_score = calculate_tfidf_similarity(STRONG_RESUME, STRONG_JOB_DESC)

    print(f"Semantic Score: {semantic_score}%")
    print(f"TF-IDF Score:   {tfidf_score}%")
    print(f"Difference:     {abs(semantic_score - tfidf_score):.2f}%")

    if semantic_score > 60 and tfidf_score > 60:
        print("\n✅ Both methods correctly identified strong match")
    else:
        print("\n⚠️  Scores lower than expected for strong match")


def test_weak_match():
    """Test 3: Weak match - different domain."""
    print_separator("TEST 3: Weak Match (Frontend Resume vs DS Job)")

    semantic_score = calculate_semantic_similarity(WEAK_RESUME, STRONG_JOB_DESC)
    tfidf_score = calculate_tfidf_similarity(WEAK_RESUME, STRONG_JOB_DESC)

    print(f"Semantic Score: {semantic_score}%")
    print(f"TF-IDF Score:   {tfidf_score}%")
    print(f"Difference:     {abs(semantic_score - tfidf_score):.2f}%")

    if semantic_score < 50 and tfidf_score < 50:
        print("\n✅ Both methods correctly identified weak match")
    else:
        print("\n⚠️  Scores higher than expected for weak match")


def test_semantic_vs_keyword():
    """Test 4: Semantic understanding vs pure keyword matching."""
    print_separator("TEST 4: Semantic vs Keyword (Paraphrased Resume)")

    semantic_score = calculate_semantic_similarity(SEMANTIC_RESUME, STRONG_JOB_DESC)
    tfidf_score = calculate_tfidf_similarity(SEMANTIC_RESUME, STRONG_JOB_DESC)

    print(f"Semantic Score: {semantic_score}%")
    print(f"TF-IDF Score:   {tfidf_score}%")
    print(f"Difference:     {abs(semantic_score - tfidf_score):.2f}%")

    print("\nNote: Semantic score should be HIGHER here because:")
    print("- Resume uses paraphrased language")
    print("- Meaning is similar but keywords differ")
    print("- Examples: 'neural networks' vs 'deep learning',")
    print("  'predictive algorithms' vs 'machine learning'")

    if semantic_score > tfidf_score:
        print("\n✅ Semantic matching captured meaning better than keyword matching")
    else:
        print("\n⚠️  Semantic score not significantly higher (may still be valid)")


def test_edge_cases():
    """Test 5: Edge cases - empty strings, None, etc."""
    print_separator("TEST 5: Edge Cases (Empty/None Inputs)")

    test_cases = [
        ("Empty resume", "", STRONG_JOB_DESC),
        ("Empty job desc", STRONG_RESUME, ""),
        ("Both empty", "", ""),
    ]

    all_passed = True
    for name, resume, job_desc in test_cases:
        try:
            semantic_score = calculate_semantic_similarity(resume, job_desc)
            tfidf_score = calculate_tfidf_similarity(resume, job_desc)

            if semantic_score == 0.0 and tfidf_score == 0.0:
                print(f"✅ {name}: Handled correctly (returned 0.0)")
            else:
                print(f"⚠️  {name}: semantic={semantic_score}, tfidf={tfidf_score}")
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: Raised exception: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All edge cases handled correctly")


def test_skill_gap_analysis():
    """Test 6: Skill gap analysis."""
    print_separator("TEST 6: Skill Gap Analysis")

    resume_skills = [
        'Python', 'Machine Learning', 'TensorFlow', 'SQL',
        'Pandas', 'NumPy', 'Matplotlib'
    ]
    required_skills = [
        'python', 'machine learning', 'tensorflow', 'sql',
        'pandas', 'numpy', 'matplotlib', 'deep learning'
    ]
    preferred_skills = ['pytorch', 'aws', 'docker', 'git']

    result = analyze_skill_match(resume_skills, required_skills, preferred_skills)

    print(f"Matched Required: {result['matched_required']}")
    print(f"Missing Required: {result['missing_required']}")
    print(f"Required Match %: {result['required_match_percentage']}%")
    print(f"\nMatched Preferred: {result['matched_preferred']}")
    print(f"Missing Preferred: {result['missing_preferred']}")

    if len(result['missing_required']) == 1 and 'Deep Learning' in result['missing_required']:
        print("\n✅ Skill gap analysis working correctly")
    else:
        print("\n⚠️  Unexpected skill gap results")


def test_hybrid_scoring():
    """Test 7: Hybrid scoring calculation."""
    print_separator("TEST 7: Hybrid Scoring")

    semantic_score = 75.0
    tfidf_score = 65.0
    skill_analysis = {
        'matched_required': ['Python', 'SQL', 'Pandas'],
        'missing_required': ['TensorFlow'],
        'matched_preferred': ['AWS'],
        'missing_preferred': ['Docker', 'Git'],
        'required_match_percentage': 75.0,
    }

    hybrid_score = calculate_hybrid_score(semantic_score, tfidf_score, skill_analysis)

    print(f"Semantic Score: {semantic_score}% (weight: 35%)")
    print(f"TF-IDF Score:   {tfidf_score}% (weight: 30%)")
    print(f"Required Match: {skill_analysis['required_match_percentage']}% (weight: 25%)")
    print(f"Preferred (calculated internally) (weight: 10%)")
    print(f"\nHybrid Score: {hybrid_score}%")

    expected_range = (65, 80)
    if expected_range[0] <= hybrid_score <= expected_range[1]:
        print(f"\n✅ Hybrid score within expected range {expected_range}")
    else:
        print(f"\n⚠️  Hybrid score outside expected range {expected_range}")


def test_full_integration():
    """Test 8: Full integration test using match_resume_to_job."""
    print_separator("TEST 8: Full Integration (Resume → Job Matching)")

    # Simulate parsed resume with extracted skills
    resume_skills = {
        'technical': [
            'Python', 'Machine Learning', 'TensorFlow', 'SQL',
            'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'NLP'
        ],
        'soft': ['Problem Solving', 'Communication']
    }

    result = match_resume_to_job(STRONG_RESUME, resume_skills, 'data_scientist')

    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"Job: {result['job_title']} at {result['job_company']}")
    print(f"\nScores:")
    print(f"  Semantic:  {result['semantic_score']}%")
    print(f"  TF-IDF:    {result['tfidf_score']}%")
    print(f"  Hybrid:    {result['hybrid_score']}%")
    print(f"  Overall:   {result['overall_score']}%")
    print(f"\nLabel: {result['score_label']['emoji']} {result['score_label']['label']}")
    print(f"Model: {result['model_used']}")
    print(f"Is Hybrid: {result['is_hybrid']}")

    print(f"\nSkill Analysis:")
    print(f"  Required Match: {result['skill_analysis']['required_match_percentage']}%")
    print(f"  Matched Required: {len(result['skill_analysis']['matched_required'])}")
    print(f"  Missing Required: {len(result['skill_analysis']['missing_required'])}")

    if result['overall_score'] > 60:
        print("\n✅ Full integration test passed - strong match detected")
    else:
        print("\n⚠️  Overall score lower than expected for strong match")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "█"*70)
    print("  AI RESUME ANALYZER - SEMANTIC MATCHING TEST SUITE")
    print("█"*70)

    try:
        test_model_info()
        test_strong_match()
        test_weak_match()
        test_semantic_vs_keyword()
        test_edge_cases()
        test_skill_gap_analysis()
        test_hybrid_scoring()
        test_full_integration()

        print_separator("TEST SUITE COMPLETE")
        print("\n✅ All tests executed. Review results above.\n")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
