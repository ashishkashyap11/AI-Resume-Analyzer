"""
Semantic Matching Module
Uses sentence-transformers to calculate semantic similarity between resume and job descriptions.

This provides meaning-based matching that complements keyword-based TF-IDF matching.

Author: Ashish Kashyap
Project: AI Resume Analyzer - EduVitae Services
Model: all-MiniLM-L6-v2 (lightweight, fast, 384-dimensional embeddings)
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ──────────────────────────────────────────────────────────────
# Model Loading (Singleton Pattern)
# ──────────────────────────────────────────────────────────────

_model = None


def get_semantic_model():
    """
    Load and cache the sentence-transformer model.
    Uses singleton pattern to avoid loading model multiple times.

    Model: all-MiniLM-L6-v2
    - Size: ~80MB
    - Speed: Fast on CPU
    - Dimensions: 384
    - Use case: General-purpose semantic similarity
    """
    global _model
    if _model is None:
        print("Loading semantic similarity model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")
    return _model


# ──────────────────────────────────────────────────────────────
# Semantic Similarity Calculation
# ──────────────────────────────────────────────────────────────

def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using sentence embeddings.

    Args:
        text1 (str): First text (e.g., resume)
        text2 (str): Second text (e.g., job description)

    Returns:
        float: Similarity score (0-100)

    This approach:
    - Captures meaning and context, not just keyword overlap
    - Handles paraphrasing and synonyms
    - More robust to different writing styles
    """
    # Handle empty or None inputs
    if not text1 or not text2:
        return 0.0

    text1 = str(text1).strip()
    text2 = str(text2).strip()

    if not text1 or not text2:
        return 0.0

    try:
        # Load model
        model = get_semantic_model()

        # Generate embeddings
        embeddings = model.encode([text1, text2])

        # Calculate cosine similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

        # Convert to percentage and ensure it's in valid range
        score = float(similarity) * 100
        score = max(0.0, min(100.0, score))

        return round(score, 2)

    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0


def calculate_semantic_similarity_batch(resume_text, job_descriptions):
    """
    Calculate semantic similarity between one resume and multiple job descriptions.
    More efficient than calling calculate_semantic_similarity() multiple times.

    Args:
        resume_text (str): Resume text
        job_descriptions (list): List of job description strings

    Returns:
        list: List of similarity scores (0-100)
    """
    if not resume_text or not job_descriptions:
        return [0.0] * len(job_descriptions)

    try:
        model = get_semantic_model()

        # Encode all texts at once (more efficient)
        all_texts = [resume_text] + list(job_descriptions)
        embeddings = model.encode(all_texts)

        # Calculate similarity between resume and each job description
        resume_embedding = embeddings[0].reshape(1, -1)
        job_embeddings = embeddings[1:]

        similarities = []
        for job_embedding in job_embeddings:
            similarity = cosine_similarity(
                resume_embedding,
                job_embedding.reshape(1, -1)
            )[0][0]
            score = float(similarity) * 100
            score = max(0.0, min(100.0, score))
            similarities.append(round(score, 2))

        return similarities

    except Exception as e:
        print(f"Error in batch semantic similarity: {e}")
        return [0.0] * len(job_descriptions)


# ──────────────────────────────────────────────────────────────
# Model Information
# ──────────────────────────────────────────────────────────────

def get_model_info():
    """
    Return information about the semantic matching model.
    Useful for transparency and debugging.
    """
    return {
        'model_name': 'all-MiniLM-L6-v2',
        'model_size': '~80MB',
        'embedding_dimensions': 384,
        'provider': 'sentence-transformers',
        'huggingface_url': 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
        'use_case': 'General-purpose semantic similarity',
        'speed': 'Fast (CPU-friendly)',
        'description': 'Lightweight sentence embedding model for semantic similarity tasks'
    }
