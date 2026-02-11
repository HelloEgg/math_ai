import os
import hashlib
import json
import base64
import io
import uuid
import re
import requests
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont

from config import Config
from models import (
    db, MathProblem, MathProblemSummary, MathProblemDeep, MathProblemOriginal,
    EnglishProblem, ScienceProblem, SocialScienceProblem, KoreanProblem,
    EnglishProblemSummary, ScienceProblemSummary, SocialScienceProblemSummary, KoreanProblemSummary,
    EnglishProblemDeep, ScienceProblemDeep, SocialScienceProblemDeep, KoreanProblemDeep
)


def create_app(config_class=Config):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)

    # Enable CORS for all routes
    CORS(app, resources={
        r"/*": {
            "origins": "*",  # Allow all origins (change to specific domains in production)
            "methods": ["GET", "POST", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Create upload directories - General
    os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    # Create upload directories - Summary
    os.makedirs(app.config['IMAGE_FOLDER_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_SUMMARY'], exist_ok=True)
    # Create upload directories - Deep
    os.makedirs(app.config['IMAGE_FOLDER_DEEP'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_DEEP'], exist_ok=True)
    # Create upload directories - Original
    os.makedirs(app.config['IMAGE_FOLDER_ORIGINAL'], exist_ok=True)
    # Create upload directories - English (base, summary, deep)
    os.makedirs(app.config['IMAGE_FOLDER_ENGLISH'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_ENGLISH_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_ENGLISH_DEEP'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_ENGLISH'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_ENGLISH_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_ENGLISH_DEEP'], exist_ok=True)
    # Create upload directories - Science (base, summary, deep)
    os.makedirs(app.config['IMAGE_FOLDER_SCIENCE'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_SCIENCE_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_SCIENCE_DEEP'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_SCIENCE'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_SCIENCE_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_SCIENCE_DEEP'], exist_ok=True)
    # Create upload directories - Social Science (base, summary, deep)
    os.makedirs(app.config['IMAGE_FOLDER_SOCIAL_SCIENCE'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_SOCIAL_SCIENCE_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_SOCIAL_SCIENCE_DEEP'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_SOCIAL_SCIENCE'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_SOCIAL_SCIENCE_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_SOCIAL_SCIENCE_DEEP'], exist_ok=True)
    # Create upload directories - Korean (base, summary, deep)
    os.makedirs(app.config['IMAGE_FOLDER_KOREAN'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_KOREAN_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER_KOREAN_DEEP'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_KOREAN'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_KOREAN_SUMMARY'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER_KOREAN_DEEP'], exist_ok=True)
    # Create upload directories - Math Twin
    os.makedirs(app.config['IMAGE_FOLDER_TWIN'], exist_ok=True)

    # Create database tables
    with app.app_context():
        db.create_all()

        # Migrate existing tables to add latex_string column if it doesn't exist
        migrate_add_latex_string_column(app)

    return app


def migrate_add_latex_string_column(app):
    """
    Add missing columns to existing tables.
    This handles migration for existing databases.
    """
    from sqlalchemy import text, inspect

    with app.app_context():
        inspector = inspect(db.engine)

        # latex_string column for all tables
        tables_for_latex = ['math_problems', 'math_problems_summary', 'math_problems_deep', 'math_problems_original']
        for table_name in tables_for_latex:
            if table_name not in inspector.get_table_names():
                continue
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            if 'latex_string' not in columns:
                print(f"Migrating {table_name}: adding latex_string column...")
                try:
                    db.session.execute(text(f'ALTER TABLE {table_name} ADD COLUMN latex_string TEXT'))
                    db.session.commit()
                    print(f"  Successfully added latex_string column to {table_name}")
                except Exception as e:
                    db.session.rollback()
                    print(f"  Warning: Could not add latex_string column to {table_name}: {e}")

        # Additional columns for math_problems_original
        if 'math_problems_original' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('math_problems_original')]

            migrations = {
                'image_url': 'ALTER TABLE math_problems_original ADD COLUMN image_url VARCHAR(2048)',
                'answer': 'ALTER TABLE math_problems_original ADD COLUMN answer TEXT',
                'feature': 'ALTER TABLE math_problems_original ADD COLUMN feature TEXT',
            }

            for col_name, sql in migrations.items():
                if col_name not in columns:
                    print(f"Migrating math_problems_original: adding {col_name} column...")
                    try:
                        db.session.execute(text(sql))
                        db.session.commit()
                        print(f"  Successfully added {col_name} column")
                    except Exception as e:
                        db.session.rollback()
                        print(f"  Warning: Could not add {col_name} column: {e}")

        # audio_path column for subject tables
        tables_for_audio = [
            'english_problems', 'english_problems_summary', 'english_problems_deep',
            'science_problems', 'science_problems_summary', 'science_problems_deep',
            'social_science_problems', 'social_science_problems_summary', 'social_science_problems_deep',
            'korean_problems', 'korean_problems_summary', 'korean_problems_deep',
        ]
        for table_name in tables_for_audio:
            if table_name not in inspector.get_table_names():
                continue
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            if 'audio_path' not in columns:
                print(f"Migrating {table_name}: adding audio_path column...")
                try:
                    db.session.execute(text(f'ALTER TABLE {table_name} ADD COLUMN audio_path VARCHAR(512)'))
                    db.session.commit()
                    print(f"  Successfully added audio_path column to {table_name}")
                except Exception as e:
                    db.session.rollback()
                    print(f"  Warning: Could not add audio_path column to {table_name}: {e}")


app = create_app()


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def compute_image_hash(file_data):
    """Compute SHA-256 hash of image data for exact matching."""
    return hashlib.sha256(file_data).hexdigest()


def extract_latex_from_image(image_data, api_key):
    """
    Extract math problem text as LaTeX from an image using Gemini Vision API.

    Args:
        image_data: Image data as bytes
        api_key: Gemini API key

    Returns:
        Extracted LaTeX string, or None if extraction fails
    """
    try:
        # Determine mime type from image data
        img = Image.open(io.BytesIO(image_data))
        mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
        if mime_type == "image/jpeg":
            mime_type = "image/jpeg"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """Extract ALL text and mathematical content from this math problem image.

Output the complete problem text in a normalized format:
1. Convert all mathematical expressions to LaTeX format
2. Include ALL text, numbers, equations, and choices if present
3. Preserve the problem structure (question text, equations, choices)
4. Remove any formatting that doesn't affect mathematical meaning

Output ONLY the extracted text/LaTeX, nothing else. No explanations or comments.
If there are multiple choice options, include them with their labels (①②③④⑤ or 1,2,3,4,5 etc.)"""

        image_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode('utf-8')
        }

        response = model.generate_content([prompt, image_part])
        latex_string = response.text.strip()

        print(f"OCR extracted: {latex_string[:200]}..." if len(latex_string) > 200 else f"OCR extracted: {latex_string}")
        return latex_string

    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return None


def extract_text_from_image(image_data, api_key):
    """
    Extract plain text from an image using Gemini Vision API.
    Used for non-math subjects (English, science, social_science, Korean).

    Args:
        image_data: Image data as bytes
        api_key: Gemini API key

    Returns:
        Extracted text string, or None if extraction fails
    """
    try:
        # Determine mime type from image data
        img = Image.open(io.BytesIO(image_data))
        mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
        if mime_type == "image/jpeg":
            mime_type = "image/jpeg"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """Extract ALL text content from this image exactly as it appears.

Output the complete text in a normalized format:
1. Include ALL text, numbers, and choices if present
2. Preserve the content structure (question text, passages, choices)
3. If there are multiple choice options, include them with their labels (①②③④⑤ or 1,2,3,4,5 or A,B,C,D,E etc.)
4. Maintain paragraph breaks with newlines

Output ONLY the extracted text, nothing else. No explanations or comments."""

        image_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode('utf-8')
        }

        response = model.generate_content([prompt, image_part])
        text_string = response.text.strip()

        print(f"Text OCR extracted: {text_string[:200]}..." if len(text_string) > 200 else f"Text OCR extracted: {text_string}")
        return text_string

    except Exception as e:
        print(f"Text OCR extraction failed: {e}")
        return None


def normalize_text(text_string):
    """
    Normalize a text string for comparison.
    Removes extra whitespace, normalizes punctuation.

    Args:
        text_string: Text string to normalize

    Returns:
        Normalized string
    """
    if not text_string:
        return ""

    # Convert to lowercase for comparison
    s = text_string.lower()

    # Normalize whitespace (multiple spaces/newlines to single space)
    s = re.sub(r'\s+', ' ', s)

    # Remove leading/trailing whitespace
    s = s.strip()

    # Normalize common punctuation variations
    s = s.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    s = s.replace('–', '-').replace('—', '-')

    return s


def calculate_text_similarity(text1, text2):
    """
    Calculate similarity ratio between two text strings.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity ratio between 0 and 1
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    if not norm1 or not norm2:
        return 0.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def find_similar_text_problem(text_string, model_class, similarity_threshold=0.85):
    """
    Find a problem with similar text content in the database.
    Used for non-math subjects (English, science, social_science, Korean).

    Args:
        text_string: The text string to search for
        model_class: The database model class to search
        similarity_threshold: Minimum similarity ratio to consider a match (default 0.85)

    Returns:
        The matching problem record, or None if no match found
    """
    if not text_string:
        return None

    # Get all problems with latex_string (which stores text for non-math subjects)
    problems = model_class.query.filter(model_class.latex_string.isnot(None)).all()

    best_match = None
    best_similarity = 0.0

    for problem in problems:
        similarity = calculate_text_similarity(text_string, problem.latex_string)
        if similarity > best_similarity and similarity >= similarity_threshold:
            best_similarity = similarity
            best_match = problem

    if best_match:
        print(f"Found similar text problem with {best_similarity:.2%} similarity: {best_match.id}")

    return best_match


def normalize_latex(latex_string):
    """
    Normalize a LaTeX string for comparison.
    Removes whitespace variations, normalizes common LaTeX patterns.

    Args:
        latex_string: LaTeX string to normalize

    Returns:
        Normalized string
    """
    if not latex_string:
        return ""

    # Convert to lowercase for comparison
    s = latex_string.lower()

    # Remove all whitespace
    s = re.sub(r'\s+', '', s)

    # Normalize common LaTeX variations
    s = s.replace('\\left(', '(').replace('\\right)', ')')
    s = s.replace('\\left[', '[').replace('\\right]', ']')
    s = s.replace('\\left{', '{').replace('\\right}', '}')
    s = s.replace('\\cdot', '*').replace('\\times', '*')
    s = s.replace('\\div', '/')

    # Remove common LaTeX commands that don't affect meaning
    s = re.sub(r'\\(displaystyle|textstyle|scriptstyle|scriptscriptstyle)', '', s)
    s = re.sub(r'\\(text|mathrm|mathbf|mathit|mathsf|mathtt|mathcal)\{([^}]*)\}', r'\2', s)

    return s


def calculate_latex_similarity(latex1, latex2):
    """
    Calculate similarity ratio between two LaTeX strings.

    Args:
        latex1: First LaTeX string
        latex2: Second LaTeX string

    Returns:
        Similarity ratio between 0 and 1
    """
    norm1 = normalize_latex(latex1)
    norm2 = normalize_latex(latex2)

    if not norm1 or not norm2:
        return 0.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def find_similar_problem(latex_string, model_class, similarity_threshold=0.85):
    """
    Find a problem with similar LaTeX content in the database.

    Args:
        latex_string: The LaTeX string to search for
        model_class: The database model class to search (MathProblem, MathProblemSummary, MathProblemDeep)
        similarity_threshold: Minimum similarity ratio to consider a match (default 0.85)

    Returns:
        The matching problem record, or None if no match found
    """
    if not latex_string:
        return None

    # Get all problems with latex_string
    problems = model_class.query.filter(model_class.latex_string.isnot(None)).all()

    best_match = None
    best_similarity = 0.0

    for problem in problems:
        similarity = calculate_latex_similarity(latex_string, problem.latex_string)
        if similarity > best_similarity and similarity >= similarity_threshold:
            best_similarity = similarity
            best_match = problem

    if best_match:
        print(f"Found similar problem with {best_similarity:.2%} similarity: {best_match.id}")

    return best_match


def save_file(file, folder, prefix):
    """Save uploaded file and return the path."""
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    new_filename = f"{prefix}.{ext}" if ext else prefix
    filepath = os.path.join(folder, new_filename)
    file.seek(0)
    file.save(filepath)
    return filepath


# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Math Problems Server is running'})


@app.route('/search', methods=['POST'])
def search_problem():
    """
    Search for a math problem by image using OCR-based content matching.

    Expects: multipart/form-data with 'image' file
    Returns: JSON with 'uuid' (string or null), 'match_type' ('exact', 'similar', or null), 'similarity' (0-1)
    """
    # Validate request has image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400

    # Read image data and compute hash
    image_data = image_file.read()

    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400

    image_hash = compute_image_hash(image_data)

    # Step 1: Try exact image hash match first (fastest)
    problem = MathProblem.query.filter_by(image_hash=image_hash).first()

    if problem:
        return jsonify({
            'uuid': problem.id,
            'match_type': 'exact',
            'similarity': 1.0
        })

    # Step 2: Extract LaTeX from image using OCR and find similar problems
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

        if latex_string:
            # Find similar problem using fuzzy matching
            similar_problem = find_similar_problem(latex_string, MathProblem, similarity_threshold=0.85)

            if similar_problem:
                similarity = calculate_latex_similarity(latex_string, similar_problem.latex_string)
                return jsonify({
                    'uuid': similar_problem.id,
                    'match_type': 'similar',
                    'similarity': round(similarity, 4)
                })

    return jsonify({
        'uuid': None,
        'match_type': None,
        'similarity': 0.0
    })


@app.route('/problems', methods=['POST'])
def create_problem():
    """
    Create a new math problem.

    Expects: multipart/form-data with:
        - 'image': math problem image file
        - 'solution_latex': LaTeX solution (text field)
        - 'audio': audio explanation file

    Returns: JSON with created problem's 'uuid'
    """
    # Validate required fields
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    if 'solution_latex' not in request.form:
        return jsonify({'error': 'No solution_latex provided'}), 400

    image_file = request.files['image']
    audio_file = request.files['audio']
    solution_latex = request.form['solution_latex']

    # Validate image
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400

    # Validate audio
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400

    # Validate solution_latex is not empty
    if not solution_latex.strip():
        return jsonify({'error': 'solution_latex cannot be empty'}), 400

    # Read image data and compute hash
    image_data = image_file.read()

    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400

    image_hash = compute_image_hash(image_data)

    # Check if problem with same image already exists
    existing = MathProblem.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({
            'error': 'A problem with this exact image already exists',
            'existing_uuid': existing.id
        }), 409

    # Extract LaTeX from image using OCR (for content-based search)
    latex_string = None
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

    # Read audio data to check size
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    # Create new problem entry
    problem = MathProblem(
        image_hash=image_hash,
        solution_latex=solution_latex,
        latex_string=latex_string,  # OCR extracted question text
        image_path='',  # Will be updated after saving
        audio_path=''   # Will be updated after saving
    )
    db.session.add(problem)
    db.session.flush()  # Get the generated UUID

    # Save files using UUID as prefix
    image_file.seek(0)
    image_path = save_file(image_file, app.config['IMAGE_FOLDER'], problem.id)

    # Reset audio file position and save
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER'], problem.id)

    # Update paths
    problem.image_path = image_path
    problem.audio_path = audio_path

    db.session.commit()

    return jsonify({
        'uuid': problem.id,
        'latex_string': latex_string,
        'message': 'Math problem created successfully'
    }), 201


@app.route('/problems/<uuid>', methods=['GET'])
def get_problem(uuid):
    """
    Get a math problem by UUID.

    Returns: JSON with 'solution_latex' and 'audio_url'
    """
    problem = MathProblem.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    return jsonify({
        'uuid': problem.id,
        'solution_latex': problem.solution_latex,
        'audio_url': f'/problems/{problem.id}/audio',
        'image_url': f'/problems/{problem.id}/image',
        'created_at': problem.created_at.isoformat() if problem.created_at else None
    })


@app.route('/problems/<uuid>/audio', methods=['GET'])
def get_problem_audio(uuid):
    """
    Get the audio file for a math problem.

    Returns: Audio file
    """
    problem = MathProblem.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    if not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404

    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems/<uuid>/image', methods=['GET'])
def get_problem_image(uuid):
    """
    Get the image file for a math problem.

    Returns: Image file
    """
    problem = MathProblem.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    if not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404

    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems', methods=['GET'])
def list_problems():
    """
    List all math problems.

    Query params:
        - page: page number (default 1)
        - per_page: items per page (default 20, max 100)

    Returns: JSON with paginated list of problems
    """
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    pagination = MathProblem.query.order_by(
        MathProblem.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    problems = [{
        'uuid': p.id,
        'image_url': f'/problems/{p.id}/image',
        'created_at': p.created_at.isoformat() if p.created_at else None
    } for p in pagination.items]

    return jsonify({
        'problems': problems,
        'total': pagination.total,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'pages': pagination.pages
    })


@app.route('/problems/<uuid>', methods=['DELETE'])
def delete_problem(uuid):
    """
    Delete a math problem by UUID.

    Returns: JSON with success message
    """
    problem = MathProblem.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    # Delete files
    if os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)

    # Delete from database
    db.session.delete(problem)
    db.session.commit()

    return jsonify({'message': 'Problem deleted successfully'})


# =============================================================================
# Original Endpoints (JSON with image URLs, no audio)
# =============================================================================

@app.route('/search_original', methods=['POST'])
def search_problem_original():
    """
    Search for an original math problem by image URL.
    Downloads the image, then searches DB by hash and OCR-based fuzzy matching.
    If found, returns the stored solution, answer, and feature.

    Expects JSON body:
    {
        "image_url": "https://example.com/problem.png",
        "feature": "something"
    }

    Returns:
        - If found: uuid, solution_latex, answer, feature, image_url, match_type, similarity
        - If not found: uuid=null
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    image_url = data.get('image_url')
    feature = data.get('feature')
    if not image_url:
        return jsonify({'error': 'No image_url provided'}), 400

    # Download image from URL
    image_data = download_image_from_url(image_url)
    if not image_data:
        return jsonify({'error': 'Failed to download image from URL'}), 400

    image_hash = compute_image_hash(image_data)

    # Step 1: Try exact image hash match first (fastest)
    problem = MathProblemOriginal.query.filter_by(image_hash=image_hash).first()

    if problem:
        return jsonify({
            'uuid': problem.id,
            'solution_latex': problem.solution_latex,
            'answer': problem.answer,
            'feature': problem.feature,
            'image_url': problem.image_url,
            'match_type': 'exact',
            'similarity': 1.0
        })

    # Step 2: Extract LaTeX from image using OCR and find similar problems
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

        if latex_string:
            similar_problem = find_similar_problem(latex_string, MathProblemOriginal, similarity_threshold=0.85)

            if similar_problem:
                similarity = calculate_latex_similarity(latex_string, similar_problem.latex_string)
                return jsonify({
                    'uuid': similar_problem.id,
                    'solution_latex': similar_problem.solution_latex,
                    'answer': similar_problem.answer,
                    'feature': similar_problem.feature,
                    'image_url': similar_problem.image_url,
                    'match_type': 'similar',
                    'similarity': round(similarity, 4)
                })

    return jsonify({
        'uuid': None,
        'solution_latex': None,
        'answer': None,
        'feature': None,
        'image_url': None,
        'match_type': None,
        'similarity': 0.0
    })


@app.route('/register_original', methods=['POST'])
def register_problem_original():
    """
    Register a new original math problem.
    Downloads the image from URL and stores solution + answer + feature in DB.

    Expects JSON body:
    {
        "image_url": "https://example.com/problem.png",
        "solution_latex": "Step-by-step solution in LaTeX...",
        "answer": "42",
        "feature": "something"
    }

    Returns: JSON with created problem's uuid
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    image_url = data.get('image_url')
    solution_latex = data.get('solution_latex')
    answer = data.get('answer')
    feature = data.get('feature')

    if not image_url:
        return jsonify({'error': 'No image_url provided'}), 400

    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400

    # Download image from URL
    image_data = download_image_from_url(image_url)
    if not image_data:
        return jsonify({'error': 'Failed to download image from URL'}), 400

    image_hash = compute_image_hash(image_data)

    # Check if problem with same image already exists
    existing = MathProblemOriginal.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({
            'error': 'An original problem with this exact image already exists',
            'existing_uuid': existing.id
        }), 409

    # Extract LaTeX from image using OCR (for content-based search)
    latex_string = None
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

    # Save image locally
    problem_id = str(uuid.uuid4())
    ext = image_url.rsplit('.', 1)[-1].lower().split('?')[0] if '.' in image_url else 'png'
    if ext not in {'png', 'jpg', 'jpeg', 'gif', 'webp'}:
        ext = 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_ORIGINAL'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    problem = MathProblemOriginal(
        id=problem_id,
        image_hash=image_hash,
        image_url=image_url,
        image_path=image_path,
        solution_latex=str(solution_latex),
        answer=str(answer) if answer is not None else None,
        feature=str(feature) if feature is not None else None,
        latex_string=latex_string
    )
    db.session.add(problem)
    db.session.commit()

    return jsonify({
        'uuid': problem.id,
        'latex_string': latex_string,
        'message': 'Original math problem registered successfully'
    }), 201


# =============================================================================
# English Endpoints
# =============================================================================

@app.route('/search_English', methods=['POST'])
def search_problem_english():
    """
    Search for an English problem by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    problem = EnglishProblem.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'match_type': 'exact', 'similarity': 1.0})

    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, EnglishProblem, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'match_type': 'similar', 'similarity': round(similarity, 4)})

    return jsonify({'uuid': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_English', methods=['POST'])
def create_problem_english():
    """
    Create a new English problem.
    Expects: multipart/form-data with 'image' file, 'audio' file, and 'solution_latex' field
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400

    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not solution_latex.strip():
        return jsonify({'error': 'No solution_latex provided'}), 400

    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    existing = EnglishProblem.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409

    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])

    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_ENGLISH'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_ENGLISH'], problem_id)

    problem = EnglishProblem(
        id=problem_id,
        image_hash=image_hash,
        image_url=f'/problems_English/{problem_id}/image',
        image_path=image_path,
        solution_latex=solution_latex,
        audio_path=audio_path,
        answer=answer if answer else None,
        feature=feature if feature else None,
        latex_string=text_string
    )
    db.session.add(problem)
    db.session.commit()

    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'English problem created successfully'}), 201


@app.route('/problems_English/<uuid>', methods=['GET'])
def get_problem_english(uuid):
    """Get an English problem by UUID."""
    problem = EnglishProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    return jsonify({
        'uuid': problem.id,
        'solution_latex': problem.solution_latex,
        'answer': problem.answer,
        'feature': problem.feature,
        'audio_url': f'/problems_English/{problem.id}/audio',
        'image_url': f'/problems_English/{problem.id}/image',
        'created_at': problem.created_at.isoformat() if problem.created_at else None
    })


@app.route('/problems_English/<uuid>/audio', methods=['GET'])
def get_problem_english_audio(uuid):
    """Get the audio file for an English problem."""
    problem = EnglishProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_English/<uuid>/image', methods=['GET'])
def get_problem_english_image(uuid):
    """Get the image file for an English problem."""
    problem = EnglishProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_English', methods=['GET'])
def list_problems_english():
    """List all English problems with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    pagination = EnglishProblem.query.order_by(
        EnglishProblem.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    problems = [{
        'uuid': p.id,
        'image_url': f'/problems_English/{p.id}/image',
        'feature': p.feature,
        'created_at': p.created_at.isoformat() if p.created_at else None
    } for p in pagination.items]

    return jsonify({
        'problems': problems,
        'total': pagination.total,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'pages': pagination.pages
    })


@app.route('/problems_English/<uuid>', methods=['DELETE'])
def delete_problem_english(uuid):
    """Delete an English problem by UUID."""
    problem = EnglishProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)

    db.session.delete(problem)
    db.session.commit()

    return jsonify({'message': 'English problem deleted successfully'})


# =============================================================================
# Science Endpoints
# =============================================================================

@app.route('/search_science', methods=['POST'])
def search_problem_science():
    """
    Search for a science problem by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    problem = ScienceProblem.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'match_type': 'exact', 'similarity': 1.0})

    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, ScienceProblem, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'match_type': 'similar', 'similarity': round(similarity, 4)})

    return jsonify({'uuid': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_science', methods=['POST'])
def create_problem_science():
    """
    Create a new science problem.
    Expects: multipart/form-data with 'image' file, 'audio' file, and 'solution_latex' field
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400

    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not solution_latex.strip():
        return jsonify({'error': 'No solution_latex provided'}), 400

    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    existing = ScienceProblem.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409

    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])

    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_SCIENCE'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_SCIENCE'], problem_id)

    problem = ScienceProblem(
        id=problem_id,
        image_hash=image_hash,
        image_url=f'/problems_science/{problem_id}/image',
        image_path=image_path,
        solution_latex=solution_latex,
        audio_path=audio_path,
        answer=answer if answer else None,
        feature=feature if feature else None,
        latex_string=text_string
    )
    db.session.add(problem)
    db.session.commit()

    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Science problem created successfully'}), 201


@app.route('/problems_science/<uuid>', methods=['GET'])
def get_problem_science(uuid):
    """Get a science problem by UUID."""
    problem = ScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    return jsonify({
        'uuid': problem.id,
        'solution_latex': problem.solution_latex,
        'answer': problem.answer,
        'feature': problem.feature,
        'audio_url': f'/problems_science/{problem.id}/audio',
        'image_url': f'/problems_science/{problem.id}/image',
        'created_at': problem.created_at.isoformat() if problem.created_at else None
    })


@app.route('/problems_science/<uuid>/audio', methods=['GET'])
def get_problem_science_audio(uuid):
    """Get the audio file for a science problem."""
    problem = ScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_science/<uuid>/image', methods=['GET'])
def get_problem_science_image(uuid):
    """Get the image file for a science problem."""
    problem = ScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_science', methods=['GET'])
def list_problems_science():
    """List all science problems with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    pagination = ScienceProblem.query.order_by(
        ScienceProblem.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    problems = [{
        'uuid': p.id,
        'image_url': f'/problems_science/{p.id}/image',
        'feature': p.feature,
        'created_at': p.created_at.isoformat() if p.created_at else None
    } for p in pagination.items]

    return jsonify({
        'problems': problems,
        'total': pagination.total,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'pages': pagination.pages
    })


@app.route('/problems_science/<uuid>', methods=['DELETE'])
def delete_problem_science(uuid):
    """Delete a science problem by UUID."""
    problem = ScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)

    db.session.delete(problem)
    db.session.commit()

    return jsonify({'message': 'Science problem deleted successfully'})


# =============================================================================
# Social Science Endpoints
# =============================================================================

@app.route('/search_social_science', methods=['POST'])
def search_problem_social_science():
    """
    Search for a social science problem by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    problem = SocialScienceProblem.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'match_type': 'exact', 'similarity': 1.0})

    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, SocialScienceProblem, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'match_type': 'similar', 'similarity': round(similarity, 4)})

    return jsonify({'uuid': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_social_science', methods=['POST'])
def create_problem_social_science():
    """
    Create a new social science problem.
    Expects: multipart/form-data with 'image' file, 'audio' file, and 'solution_latex' field
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400

    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not solution_latex.strip():
        return jsonify({'error': 'No solution_latex provided'}), 400

    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    existing = SocialScienceProblem.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409

    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])

    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_SOCIAL_SCIENCE'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_SOCIAL_SCIENCE'], problem_id)

    problem = SocialScienceProblem(
        id=problem_id,
        image_hash=image_hash,
        image_url=f'/problems_social_science/{problem_id}/image',
        image_path=image_path,
        solution_latex=solution_latex,
        audio_path=audio_path,
        answer=answer if answer else None,
        feature=feature if feature else None,
        latex_string=text_string
    )
    db.session.add(problem)
    db.session.commit()

    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Social science problem created successfully'}), 201


@app.route('/problems_social_science/<uuid>', methods=['GET'])
def get_problem_social_science(uuid):
    """Get a social science problem by UUID."""
    problem = SocialScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    return jsonify({
        'uuid': problem.id,
        'solution_latex': problem.solution_latex,
        'answer': problem.answer,
        'feature': problem.feature,
        'audio_url': f'/problems_social_science/{problem.id}/audio',
        'image_url': f'/problems_social_science/{problem.id}/image',
        'created_at': problem.created_at.isoformat() if problem.created_at else None
    })


@app.route('/problems_social_science/<uuid>/audio', methods=['GET'])
def get_problem_social_science_audio(uuid):
    """Get the audio file for a social science problem."""
    problem = SocialScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_social_science/<uuid>/image', methods=['GET'])
def get_problem_social_science_image(uuid):
    """Get the image file for a social science problem."""
    problem = SocialScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_social_science', methods=['GET'])
def list_problems_social_science():
    """List all social science problems with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    pagination = SocialScienceProblem.query.order_by(
        SocialScienceProblem.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    problems = [{
        'uuid': p.id,
        'image_url': f'/problems_social_science/{p.id}/image',
        'feature': p.feature,
        'created_at': p.created_at.isoformat() if p.created_at else None
    } for p in pagination.items]

    return jsonify({
        'problems': problems,
        'total': pagination.total,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'pages': pagination.pages
    })


@app.route('/problems_social_science/<uuid>', methods=['DELETE'])
def delete_problem_social_science(uuid):
    """Delete a social science problem by UUID."""
    problem = SocialScienceProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)

    db.session.delete(problem)
    db.session.commit()

    return jsonify({'message': 'Social science problem deleted successfully'})


# =============================================================================
# Korean Endpoints
# =============================================================================

@app.route('/search_Korean', methods=['POST'])
def search_problem_korean():
    """
    Search for a Korean problem by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    problem = KoreanProblem.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'match_type': 'exact', 'similarity': 1.0})

    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, KoreanProblem, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'match_type': 'similar', 'similarity': round(similarity, 4)})

    return jsonify({'uuid': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_Korean', methods=['POST'])
def create_problem_korean():
    """
    Create a new Korean problem.
    Expects: multipart/form-data with 'image' file, 'audio' file, and 'solution_latex' field
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400

    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not solution_latex.strip():
        return jsonify({'error': 'No solution_latex provided'}), 400

    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    existing = KoreanProblem.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409

    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])

    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_KOREAN'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_KOREAN'], problem_id)

    problem = KoreanProblem(
        id=problem_id,
        image_hash=image_hash,
        image_url=f'/problems_Korean/{problem_id}/image',
        image_path=image_path,
        solution_latex=solution_latex,
        audio_path=audio_path,
        answer=answer if answer else None,
        feature=feature if feature else None,
        latex_string=text_string
    )
    db.session.add(problem)
    db.session.commit()

    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Korean problem created successfully'}), 201


@app.route('/problems_Korean/<uuid>', methods=['GET'])
def get_problem_korean(uuid):
    """Get a Korean problem by UUID."""
    problem = KoreanProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    return jsonify({
        'uuid': problem.id,
        'solution_latex': problem.solution_latex,
        'answer': problem.answer,
        'feature': problem.feature,
        'audio_url': f'/problems_Korean/{problem.id}/audio',
        'image_url': f'/problems_Korean/{problem.id}/image',
        'created_at': problem.created_at.isoformat() if problem.created_at else None
    })


@app.route('/problems_Korean/<uuid>/audio', methods=['GET'])
def get_problem_korean_audio(uuid):
    """Get the audio file for a Korean problem."""
    problem = KoreanProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_Korean/<uuid>/image', methods=['GET'])
def get_problem_korean_image(uuid):
    """Get the image file for a Korean problem."""
    problem = KoreanProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_Korean', methods=['GET'])
def list_problems_korean():
    """List all Korean problems with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    pagination = KoreanProblem.query.order_by(
        KoreanProblem.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    problems = [{
        'uuid': p.id,
        'image_url': f'/problems_Korean/{p.id}/image',
        'feature': p.feature,
        'created_at': p.created_at.isoformat() if p.created_at else None
    } for p in pagination.items]

    return jsonify({
        'problems': problems,
        'total': pagination.total,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'pages': pagination.pages
    })


@app.route('/problems_Korean/<uuid>', methods=['DELETE'])
def delete_problem_korean(uuid):
    """Delete a Korean problem by UUID."""
    problem = KoreanProblem.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404

    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)

    db.session.delete(problem)
    db.session.commit()

    return jsonify({'message': 'Korean problem deleted successfully'})


# =============================================================================
# English Summary Endpoints
# =============================================================================

@app.route('/search_English_summary', methods=['POST'])
def search_problem_english_summary():
    """
    Search for an English problem summary by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = EnglishProblemSummary.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_English_summary/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, EnglishProblemSummary, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_English_summary/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_English_summary', methods=['POST'])
def create_problem_english_summary():
    """
    Create a new English problem summary.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = EnglishProblemSummary.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_ENGLISH_SUMMARY'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_ENGLISH_SUMMARY'], problem_id)
    problem = EnglishProblemSummary(id=problem_id, image_hash=image_hash, image_url=f'/problems_English_summary/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'English problem summary created successfully'}), 201


@app.route('/problems_English_summary/<uuid>', methods=['GET'])
def get_problem_english_summary(uuid):
    """Get an English problem summary by UUID."""
    problem = EnglishProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_English_summary/{problem.id}/audio', 'image_url': f'/problems_English_summary/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_English_summary/<uuid>/audio', methods=['GET'])
def get_problem_english_summary_audio(uuid):
    """Get the audio file for an English problem summary."""
    problem = EnglishProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_English_summary/<uuid>/image', methods=['GET'])
def get_problem_english_summary_image(uuid):
    """Get the image file for an English problem summary."""
    problem = EnglishProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_English_summary', methods=['GET'])
def list_problems_english_summary():
    """List all English problem summaries with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = EnglishProblemSummary.query.order_by(EnglishProblemSummary.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_English_summary/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_English_summary/<uuid>', methods=['DELETE'])
def delete_problem_english_summary(uuid):
    """Delete an English problem summary by UUID."""
    problem = EnglishProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'English problem summary deleted successfully'})


# =============================================================================
# English Deep Endpoints
# =============================================================================

@app.route('/search_English_deep', methods=['POST'])
def search_problem_english_deep():
    """
    Search for an English problem deep by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = EnglishProblemDeep.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_English_deep/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, EnglishProblemDeep, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_English_deep/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_English_deep', methods=['POST'])
def create_problem_english_deep():
    """
    Create a new English problem deep.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = EnglishProblemDeep.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_ENGLISH_DEEP'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_ENGLISH_DEEP'], problem_id)
    problem = EnglishProblemDeep(id=problem_id, image_hash=image_hash, image_url=f'/problems_English_deep/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'English problem deep created successfully'}), 201


@app.route('/problems_English_deep/<uuid>', methods=['GET'])
def get_problem_english_deep(uuid):
    """Get an English problem deep by UUID."""
    problem = EnglishProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_English_deep/{problem.id}/audio', 'image_url': f'/problems_English_deep/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_English_deep/<uuid>/audio', methods=['GET'])
def get_problem_english_deep_audio(uuid):
    """Get the audio file for an English problem deep."""
    problem = EnglishProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_English_deep/<uuid>/image', methods=['GET'])
def get_problem_english_deep_image(uuid):
    """Get the image file for an English problem deep."""
    problem = EnglishProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_English_deep', methods=['GET'])
def list_problems_english_deep():
    """List all English problem deeps with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = EnglishProblemDeep.query.order_by(EnglishProblemDeep.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_English_deep/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_English_deep/<uuid>', methods=['DELETE'])
def delete_problem_english_deep(uuid):
    """Delete an English problem deep by UUID."""
    problem = EnglishProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'English problem deep deleted successfully'})


# =============================================================================
# Science Summary Endpoints
# =============================================================================

@app.route('/search_science_summary', methods=['POST'])
def search_problem_science_summary():
    """
    Search for a science problem summary by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = ScienceProblemSummary.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_science_summary/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, ScienceProblemSummary, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_science_summary/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_science_summary', methods=['POST'])
def create_problem_science_summary():
    """
    Create a new science problem summary.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = ScienceProblemSummary.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_SCIENCE_SUMMARY'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_SCIENCE_SUMMARY'], problem_id)
    problem = ScienceProblemSummary(id=problem_id, image_hash=image_hash, image_url=f'/problems_science_summary/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Science problem summary created successfully'}), 201


@app.route('/problems_science_summary/<uuid>', methods=['GET'])
def get_problem_science_summary(uuid):
    """Get a science problem summary by UUID."""
    problem = ScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_science_summary/{problem.id}/audio', 'image_url': f'/problems_science_summary/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_science_summary/<uuid>/audio', methods=['GET'])
def get_problem_science_summary_audio(uuid):
    """Get the audio file for a science problem summary."""
    problem = ScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_science_summary/<uuid>/image', methods=['GET'])
def get_problem_science_summary_image(uuid):
    """Get the image file for a science problem summary."""
    problem = ScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_science_summary', methods=['GET'])
def list_problems_science_summary():
    """List all science problem summaries with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = ScienceProblemSummary.query.order_by(ScienceProblemSummary.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_science_summary/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_science_summary/<uuid>', methods=['DELETE'])
def delete_problem_science_summary(uuid):
    """Delete a science problem summary by UUID."""
    problem = ScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'Science problem summary deleted successfully'})


# =============================================================================
# Science Deep Endpoints
# =============================================================================

@app.route('/search_science_deep', methods=['POST'])
def search_problem_science_deep():
    """
    Search for a science problem deep by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = ScienceProblemDeep.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_science_deep/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, ScienceProblemDeep, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_science_deep/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_science_deep', methods=['POST'])
def create_problem_science_deep():
    """
    Create a new science problem deep.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = ScienceProblemDeep.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_SCIENCE_DEEP'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_SCIENCE_DEEP'], problem_id)
    problem = ScienceProblemDeep(id=problem_id, image_hash=image_hash, image_url=f'/problems_science_deep/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Science problem deep created successfully'}), 201


@app.route('/problems_science_deep/<uuid>', methods=['GET'])
def get_problem_science_deep(uuid):
    """Get a science problem deep by UUID."""
    problem = ScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_science_deep/{problem.id}/audio', 'image_url': f'/problems_science_deep/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_science_deep/<uuid>/audio', methods=['GET'])
def get_problem_science_deep_audio(uuid):
    """Get the audio file for a science problem deep."""
    problem = ScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_science_deep/<uuid>/image', methods=['GET'])
def get_problem_science_deep_image(uuid):
    """Get the image file for a science problem deep."""
    problem = ScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_science_deep', methods=['GET'])
def list_problems_science_deep():
    """List all science problem deeps with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = ScienceProblemDeep.query.order_by(ScienceProblemDeep.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_science_deep/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_science_deep/<uuid>', methods=['DELETE'])
def delete_problem_science_deep(uuid):
    """Delete a science problem deep by UUID."""
    problem = ScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'Science problem deep deleted successfully'})


# =============================================================================
# Social Science Summary Endpoints
# =============================================================================

@app.route('/search_social_science_summary', methods=['POST'])
def search_problem_social_science_summary():
    """
    Search for a social science problem summary by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = SocialScienceProblemSummary.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_social_science_summary/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, SocialScienceProblemSummary, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_social_science_summary/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_social_science_summary', methods=['POST'])
def create_problem_social_science_summary():
    """
    Create a new social science problem summary.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = SocialScienceProblemSummary.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_SOCIAL_SCIENCE_SUMMARY'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_SOCIAL_SCIENCE_SUMMARY'], problem_id)
    problem = SocialScienceProblemSummary(id=problem_id, image_hash=image_hash, image_url=f'/problems_social_science_summary/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Social science problem summary created successfully'}), 201


@app.route('/problems_social_science_summary/<uuid>', methods=['GET'])
def get_problem_social_science_summary(uuid):
    """Get a social science problem summary by UUID."""
    problem = SocialScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_social_science_summary/{problem.id}/audio', 'image_url': f'/problems_social_science_summary/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_social_science_summary/<uuid>/audio', methods=['GET'])
def get_problem_social_science_summary_audio(uuid):
    """Get the audio file for a social science problem summary."""
    problem = SocialScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_social_science_summary/<uuid>/image', methods=['GET'])
def get_problem_social_science_summary_image(uuid):
    """Get the image file for a social science problem summary."""
    problem = SocialScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_social_science_summary', methods=['GET'])
def list_problems_social_science_summary():
    """List all social science problem summaries with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = SocialScienceProblemSummary.query.order_by(SocialScienceProblemSummary.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_social_science_summary/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_social_science_summary/<uuid>', methods=['DELETE'])
def delete_problem_social_science_summary(uuid):
    """Delete a social science problem summary by UUID."""
    problem = SocialScienceProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'Social science problem summary deleted successfully'})


# =============================================================================
# Social Science Deep Endpoints
# =============================================================================

@app.route('/search_social_science_deep', methods=['POST'])
def search_problem_social_science_deep():
    """
    Search for a social science problem deep by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = SocialScienceProblemDeep.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_social_science_deep/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, SocialScienceProblemDeep, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_social_science_deep/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_social_science_deep', methods=['POST'])
def create_problem_social_science_deep():
    """
    Create a new social science problem deep.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = SocialScienceProblemDeep.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_SOCIAL_SCIENCE_DEEP'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_SOCIAL_SCIENCE_DEEP'], problem_id)
    problem = SocialScienceProblemDeep(id=problem_id, image_hash=image_hash, image_url=f'/problems_social_science_deep/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Social science problem deep created successfully'}), 201


@app.route('/problems_social_science_deep/<uuid>', methods=['GET'])
def get_problem_social_science_deep(uuid):
    """Get a social science problem deep by UUID."""
    problem = SocialScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_social_science_deep/{problem.id}/audio', 'image_url': f'/problems_social_science_deep/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_social_science_deep/<uuid>/audio', methods=['GET'])
def get_problem_social_science_deep_audio(uuid):
    """Get the audio file for a social science problem deep."""
    problem = SocialScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_social_science_deep/<uuid>/image', methods=['GET'])
def get_problem_social_science_deep_image(uuid):
    """Get the image file for a social science problem deep."""
    problem = SocialScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_social_science_deep', methods=['GET'])
def list_problems_social_science_deep():
    """List all social science problem deeps with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = SocialScienceProblemDeep.query.order_by(SocialScienceProblemDeep.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_social_science_deep/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_social_science_deep/<uuid>', methods=['DELETE'])
def delete_problem_social_science_deep(uuid):
    """Delete a social science problem deep by UUID."""
    problem = SocialScienceProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'Social science problem deep deleted successfully'})


# =============================================================================
# Korean Summary Endpoints
# =============================================================================

@app.route('/search_Korean_summary', methods=['POST'])
def search_problem_korean_summary():
    """
    Search for a Korean problem summary by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = KoreanProblemSummary.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_Korean_summary/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, KoreanProblemSummary, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_Korean_summary/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_Korean_summary', methods=['POST'])
def create_problem_korean_summary():
    """
    Create a new Korean problem summary.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = KoreanProblemSummary.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_KOREAN_SUMMARY'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_KOREAN_SUMMARY'], problem_id)
    problem = KoreanProblemSummary(id=problem_id, image_hash=image_hash, image_url=f'/problems_Korean_summary/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Korean problem summary created successfully'}), 201


@app.route('/problems_Korean_summary/<uuid>', methods=['GET'])
def get_problem_korean_summary(uuid):
    """Get a Korean problem summary by UUID."""
    problem = KoreanProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_Korean_summary/{problem.id}/audio', 'image_url': f'/problems_Korean_summary/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_Korean_summary/<uuid>/audio', methods=['GET'])
def get_problem_korean_summary_audio(uuid):
    """Get the audio file for a Korean problem summary."""
    problem = KoreanProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_Korean_summary/<uuid>/image', methods=['GET'])
def get_problem_korean_summary_image(uuid):
    """Get the image file for a Korean problem summary."""
    problem = KoreanProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_Korean_summary', methods=['GET'])
def list_problems_korean_summary():
    """List all Korean problem summaries with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = KoreanProblemSummary.query.order_by(KoreanProblemSummary.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_Korean_summary/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_Korean_summary/<uuid>', methods=['DELETE'])
def delete_problem_korean_summary(uuid):
    """Delete a Korean problem summary by UUID."""
    problem = KoreanProblemSummary.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'Korean problem summary deleted successfully'})


# =============================================================================
# Korean Deep Endpoints
# =============================================================================

@app.route('/search_Korean_deep', methods=['POST'])
def search_problem_korean_deep():
    """
    Search for a Korean problem deep by image using OCR-based content matching.
    Expects: multipart/form-data with 'image' file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    problem = KoreanProblemDeep.query.filter_by(image_hash=image_hash).first()
    if problem:
        return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'image_url': f'/problems_Korean_deep/{problem.id}/image', 'match_type': 'exact', 'similarity': 1.0})
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
        if text_string:
            similar_problem = find_similar_text_problem(text_string, KoreanProblemDeep, similarity_threshold=0.85)
            if similar_problem:
                similarity = calculate_text_similarity(text_string, similar_problem.latex_string)
                return jsonify({'uuid': similar_problem.id, 'solution_latex': similar_problem.solution_latex, 'answer': similar_problem.answer, 'feature': similar_problem.feature, 'image_url': f'/problems_Korean_deep/{similar_problem.id}/image', 'match_type': 'similar', 'similarity': round(similarity, 4)})
    return jsonify({'uuid': None, 'solution_latex': None, 'answer': None, 'feature': None, 'image_url': None, 'match_type': None, 'similarity': 0.0})


@app.route('/problems_Korean_deep', methods=['POST'])
def create_problem_korean_deep():
    """
    Create a new Korean problem deep.
    Expects: multipart/form-data with 'image' file, 'audio' file, 'solution_latex', and optional 'answer', 'feature'
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    image_file = request.files['image']
    audio_file = request.files['audio']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400
    solution_latex = request.form.get('solution_latex')
    answer = request.form.get('answer')
    feature = request.form.get('feature')
    if not solution_latex or not str(solution_latex).strip():
        return jsonify({'error': 'No solution_latex provided'}), 400
    image_data = image_file.read()
    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400
    image_hash = compute_image_hash(image_data)
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400
    existing = KoreanProblemDeep.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({'error': 'Problem with this image already exists', 'existing_uuid': existing.id}), 409
    text_string = None
    if app.config.get('GEMINI_API_KEY'):
        text_string = extract_text_from_image(image_data, app.config['GEMINI_API_KEY'])
    problem_id = str(uuid.uuid4())
    ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else 'png'
    image_filename = f"{problem_id}.{ext}"
    image_path = os.path.join(app.config['IMAGE_FOLDER_KOREAN_DEEP'], image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_KOREAN_DEEP'], problem_id)
    problem = KoreanProblemDeep(id=problem_id, image_hash=image_hash, image_url=f'/problems_Korean_deep/{problem_id}/image', image_path=image_path, solution_latex=str(solution_latex), audio_path=audio_path, answer=str(answer) if answer else None, feature=str(feature) if feature else None, latex_string=text_string)
    db.session.add(problem)
    db.session.commit()
    return jsonify({'uuid': problem.id, 'text_string': text_string, 'message': 'Korean problem deep created successfully'}), 201


@app.route('/problems_Korean_deep/<uuid>', methods=['GET'])
def get_problem_korean_deep(uuid):
    """Get a Korean problem deep by UUID."""
    problem = KoreanProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    return jsonify({'uuid': problem.id, 'solution_latex': problem.solution_latex, 'answer': problem.answer, 'feature': problem.feature, 'audio_url': f'/problems_Korean_deep/{problem.id}/audio', 'image_url': f'/problems_Korean_deep/{problem.id}/image', 'created_at': problem.created_at.isoformat() if problem.created_at else None})


@app.route('/problems_Korean_deep/<uuid>/audio', methods=['GET'])
def get_problem_korean_deep_audio(uuid):
    """Get the audio file for a Korean problem deep."""
    problem = KoreanProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.audio_path or not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_Korean_deep/<uuid>/image', methods=['GET'])
def get_problem_korean_deep_image(uuid):
    """Get the image file for a Korean problem deep."""
    problem = KoreanProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if not problem.image_path or not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404
    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_Korean_deep', methods=['GET'])
def list_problems_korean_deep():
    """List all Korean problem deeps with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    pagination = KoreanProblemDeep.query.order_by(KoreanProblemDeep.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    problems = [{'uuid': p.id, 'image_url': f'/problems_Korean_deep/{p.id}/image', 'feature': p.feature, 'created_at': p.created_at.isoformat() if p.created_at else None} for p in pagination.items]
    return jsonify({'problems': problems, 'total': pagination.total, 'page': pagination.page, 'per_page': pagination.per_page, 'pages': pagination.pages})


@app.route('/problems_Korean_deep/<uuid>', methods=['DELETE'])
def delete_problem_korean_deep(uuid):
    """Delete a Korean problem deep by UUID."""
    problem = KoreanProblemDeep.query.get(uuid)
    if not problem:
        return jsonify({'error': 'Problem not found'}), 404
    if problem.image_path and os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if problem.audio_path and os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)
    db.session.delete(problem)
    db.session.commit()
    return jsonify({'message': 'Korean problem deep deleted successfully'})


# =============================================================================
# Math Summary Endpoints
# =============================================================================

@app.route('/search_summary', methods=['POST'])
def search_problem_summary():
    """
    Search for a math problem summary by image using OCR-based content matching.

    Expects: multipart/form-data with 'image' file
    Returns: JSON with 'uuid' (string or null), 'match_type' ('exact', 'similar', or null), 'similarity' (0-1)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400

    image_data = image_file.read()

    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400

    image_hash = compute_image_hash(image_data)

    # Step 1: Try exact image hash match first (fastest)
    problem = MathProblemSummary.query.filter_by(image_hash=image_hash).first()

    if problem:
        return jsonify({
            'uuid': problem.id,
            'match_type': 'exact',
            'similarity': 1.0
        })

    # Step 2: Extract LaTeX from image using OCR and find similar problems
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

        if latex_string:
            # Find similar problem using fuzzy matching
            similar_problem = find_similar_problem(latex_string, MathProblemSummary, similarity_threshold=0.85)

            if similar_problem:
                similarity = calculate_latex_similarity(latex_string, similar_problem.latex_string)
                return jsonify({
                    'uuid': similar_problem.id,
                    'match_type': 'similar',
                    'similarity': round(similarity, 4)
                })

    return jsonify({
        'uuid': None,
        'match_type': None,
        'similarity': 0.0
    })


@app.route('/problems_summary', methods=['POST'])
def create_problem_summary():
    """Create a new math problem summary."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    if 'solution_latex' not in request.form:
        return jsonify({'error': 'No solution_latex provided'}), 400

    image_file = request.files['image']
    audio_file = request.files['audio']
    solution_latex = request.form['solution_latex']

    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400

    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400

    if not solution_latex.strip():
        return jsonify({'error': 'solution_latex cannot be empty'}), 400

    image_data = image_file.read()

    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400

    image_hash = compute_image_hash(image_data)

    existing = MathProblemSummary.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({
            'error': 'A problem summary with this exact image already exists',
            'existing_uuid': existing.id
        }), 409

    # Extract LaTeX from image using OCR (for content-based search)
    latex_string = None
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    problem = MathProblemSummary(
        image_hash=image_hash,
        solution_latex=solution_latex,
        latex_string=latex_string,  # OCR extracted question text
        image_path='',
        audio_path=''
    )
    db.session.add(problem)
    db.session.flush()

    image_file.seek(0)
    image_path = save_file(image_file, app.config['IMAGE_FOLDER_SUMMARY'], problem.id)

    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_SUMMARY'], problem.id)

    problem.image_path = image_path
    problem.audio_path = audio_path

    db.session.commit()

    return jsonify({
        'uuid': problem.id,
        'latex_string': latex_string,
        'message': 'Math problem summary created successfully'
    }), 201


@app.route('/problems_summary/<uuid>', methods=['GET'])
def get_problem_summary(uuid):
    """Get a math problem summary by UUID."""
    problem = MathProblemSummary.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem summary not found'}), 404

    return jsonify({
        'uuid': problem.id,
        'solution_latex': problem.solution_latex,
        'audio_url': f'/problems_summary/{problem.id}/audio',
        'image_url': f'/problems_summary/{problem.id}/image',
        'created_at': problem.created_at.isoformat() if problem.created_at else None
    })


@app.route('/problems_summary/<uuid>/audio', methods=['GET'])
def get_problem_summary_audio(uuid):
    """Get the audio file for a math problem summary."""
    problem = MathProblemSummary.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem summary not found'}), 404

    if not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404

    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_summary/<uuid>/image', methods=['GET'])
def get_problem_summary_image(uuid):
    """Get the image file for a math problem summary."""
    problem = MathProblemSummary.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem summary not found'}), 404

    if not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404

    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_summary', methods=['GET'])
def list_problems_summary():
    """List all math problem summaries."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    pagination = MathProblemSummary.query.order_by(
        MathProblemSummary.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    problems = [{
        'uuid': p.id,
        'image_url': f'/problems_summary/{p.id}/image',
        'created_at': p.created_at.isoformat() if p.created_at else None
    } for p in pagination.items]

    return jsonify({
        'problems': problems,
        'total': pagination.total,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'pages': pagination.pages
    })


@app.route('/problems_summary/<uuid>', methods=['DELETE'])
def delete_problem_summary(uuid):
    """Delete a math problem summary by UUID."""
    problem = MathProblemSummary.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Problem summary not found'}), 404

    if os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)

    db.session.delete(problem)
    db.session.commit()

    return jsonify({'message': 'Problem summary deleted successfully'})


# =============================================================================
# Deep Endpoints
# =============================================================================

@app.route('/search_deep', methods=['POST'])
def search_problem_deep():
    """
    Search for a deep math problem by image using OCR-based content matching.

    Expects: multipart/form-data with 'image' file
    Returns: JSON with 'uuid' (string or null), 'match_type' ('exact', 'similar', or null), 'similarity' (0-1)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400

    image_data = image_file.read()

    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400

    image_hash = compute_image_hash(image_data)

    # Step 1: Try exact image hash match first (fastest)
    problem = MathProblemDeep.query.filter_by(image_hash=image_hash).first()

    if problem:
        return jsonify({
            'uuid': problem.id,
            'match_type': 'exact',
            'similarity': 1.0
        })

    # Step 2: Extract LaTeX from image using OCR and find similar problems
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

        if latex_string:
            # Find similar problem using fuzzy matching
            similar_problem = find_similar_problem(latex_string, MathProblemDeep, similarity_threshold=0.85)

            if similar_problem:
                similarity = calculate_latex_similarity(latex_string, similar_problem.latex_string)
                return jsonify({
                    'uuid': similar_problem.id,
                    'match_type': 'similar',
                    'similarity': round(similarity, 4)
                })

    return jsonify({
        'uuid': None,
        'match_type': None,
        'similarity': 0.0
    })


@app.route('/problems_deep', methods=['POST'])
def create_problem_deep():
    """Create a new deep math problem."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    if 'solution_latex' not in request.form:
        return jsonify({'error': 'No solution_latex provided'}), 400

    image_file = request.files['image']
    audio_file = request.files['audio']
    solution_latex = request.form['solution_latex']

    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400

    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
        return jsonify({'error': 'Invalid audio file type'}), 400

    if not solution_latex.strip():
        return jsonify({'error': 'solution_latex cannot be empty'}), 400

    image_data = image_file.read()

    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400

    image_hash = compute_image_hash(image_data)

    existing = MathProblemDeep.query.filter_by(image_hash=image_hash).first()
    if existing:
        return jsonify({
            'error': 'A deep problem with this exact image already exists',
            'existing_uuid': existing.id
        }), 409

    # Extract LaTeX from image using OCR (for content-based search)
    latex_string = None
    if app.config.get('GEMINI_API_KEY'):
        latex_string = extract_latex_from_image(image_data, app.config['GEMINI_API_KEY'])

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    problem = MathProblemDeep(
        image_hash=image_hash,
        solution_latex=solution_latex,
        latex_string=latex_string,  # OCR extracted question text
        image_path='',
        audio_path=''
    )
    db.session.add(problem)
    db.session.flush()

    image_file.seek(0)
    image_path = save_file(image_file, app.config['IMAGE_FOLDER_DEEP'], problem.id)

    audio_file.seek(0)
    audio_path = save_file(audio_file, app.config['AUDIO_FOLDER_DEEP'], problem.id)

    problem.image_path = image_path
    problem.audio_path = audio_path

    db.session.commit()

    return jsonify({
        'uuid': problem.id,
        'latex_string': latex_string,
        'message': 'Deep math problem created successfully'
    }), 201


@app.route('/problems_deep/<uuid>', methods=['GET'])
def get_problem_deep(uuid):
    """Get a deep math problem by UUID."""
    problem = MathProblemDeep.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Deep problem not found'}), 404

    return jsonify({
        'uuid': problem.id,
        'solution_latex': problem.solution_latex,
        'audio_url': f'/problems_deep/{problem.id}/audio',
        'image_url': f'/problems_deep/{problem.id}/image',
        'created_at': problem.created_at.isoformat() if problem.created_at else None
    })


@app.route('/problems_deep/<uuid>/audio', methods=['GET'])
def get_problem_deep_audio(uuid):
    """Get the audio file for a deep math problem."""
    problem = MathProblemDeep.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Deep problem not found'}), 404

    if not os.path.exists(problem.audio_path):
        return jsonify({'error': 'Audio file not found'}), 404

    return send_file(problem.audio_path, as_attachment=False)


@app.route('/problems_deep/<uuid>/image', methods=['GET'])
def get_problem_deep_image(uuid):
    """Get the image file for a deep math problem."""
    problem = MathProblemDeep.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Deep problem not found'}), 404

    if not os.path.exists(problem.image_path):
        return jsonify({'error': 'Image file not found'}), 404

    return send_file(problem.image_path, as_attachment=False)


@app.route('/problems_deep', methods=['GET'])
def list_problems_deep():
    """List all deep math problems."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    pagination = MathProblemDeep.query.order_by(
        MathProblemDeep.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    problems = [{
        'uuid': p.id,
        'image_url': f'/problems_deep/{p.id}/image',
        'created_at': p.created_at.isoformat() if p.created_at else None
    } for p in pagination.items]

    return jsonify({
        'problems': problems,
        'total': pagination.total,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'pages': pagination.pages
    })


@app.route('/problems_deep/<uuid>', methods=['DELETE'])
def delete_problem_deep(uuid):
    """Delete a deep math problem by UUID."""
    problem = MathProblemDeep.query.get(uuid)

    if not problem:
        return jsonify({'error': 'Deep problem not found'}), 404

    if os.path.exists(problem.image_path):
        os.remove(problem.image_path)
    if os.path.exists(problem.audio_path):
        os.remove(problem.audio_path)

    db.session.delete(problem)
    db.session.commit()

    return jsonify({'message': 'Deep problem deleted successfully'})


# =============================================================================
# Math Twin Endpoint (Gemini AI) with Image Modification
# =============================================================================

@app.route('/math_twin/images/<image_uuid>', methods=['GET'])
def get_math_twin_image(image_uuid):
    """
    Get a modified math twin image by UUID.

    Query params:
        - download: if set to 'true', forces download with filename

    Returns: Image file (PNG)
    """
    # Validate UUID format
    try:
        uuid.UUID(image_uuid)
    except ValueError:
        return jsonify({'error': 'Invalid image UUID format'}), 400

    image_path = os.path.join(app.config['IMAGE_FOLDER_TWIN'], f"{image_uuid}.png")

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404

    # Check if download is requested
    download = request.args.get('download', '').lower() == 'true'

    if download:
        return send_file(
            image_path,
            as_attachment=True,
            download_name=f"math_twin_{image_uuid}.png"
        )
    else:
        return send_file(image_path, mimetype='image/png')

def fix_latex_escaping(text):
    """
    Fix LaTeX backslash escaping in JSON strings.
    Gemini often returns unescaped LaTeX like \\frac which breaks JSON parsing.

    In JSON, valid escape sequences are: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
    Any other backslash followed by a character is invalid and needs to be escaped.
    """
    # Process character by character to handle mixed escaping
    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\':
            if i + 1 < len(text):
                next_char = text[i + 1]
                # Check if already escaped (double backslash)
                if next_char == '\\':
                    # Already escaped, keep both
                    result.append('\\\\')
                    i += 2
                    continue
                # Check if it's a valid JSON escape
                elif next_char in '"\/bfnrtu':
                    # Valid JSON escape, keep as is
                    result.append('\\')
                    result.append(next_char)
                    i += 2
                    continue
                else:
                    # Invalid escape (like \frac, \(, etc.), add extra backslash
                    result.append('\\\\')
                    result.append(next_char)
                    i += 2
                    continue
            else:
                # Trailing backslash, keep it
                result.append('\\')
                i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


def extract_json_from_response(response_text):
    """Extract JSON from response, handling markdown code blocks."""
    response_text = response_text.strip()
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith('```') and not in_json:
                in_json = True
                continue
            elif line.startswith('```') and in_json:
                break
            elif in_json:
                json_lines.append(line)
        response_text = '\n'.join(json_lines)

    # Fix LaTeX escaping issues
    response_text = fix_latex_escaping(response_text)

    return response_text


def ensure_white_background(image_data):
    """
    Post-process generated image to ensure pure white background.
    Converts any near-white or light gray pixels to pure white.

    Args:
        image_data: Image data as bytes

    Returns:
        Processed image data as bytes (PNG format)
    """
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        pixels = img.load()
        width, height = img.size

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                # If pixel is light gray or near-white (all channels > 200), make it pure white
                if r > 200 and g > 200 and b > 200:
                    pixels[x, y] = (255, 255, 255)

        output = io.BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
    except Exception as e:
        print(f"White background conversion failed: {e}")
        return image_data


def generate_solution_image(api_key, solution_text):
    """
    Generate a solution-only image (no question text) using Gemini's image generation model.

    Args:
        api_key: Gemini API key
        solution_text: The solution/explanation text in LaTeX format

    Returns:
        Image data as bytes, or None if generation fails
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = f"""수학 문제의 해설(풀이)만 포함하는 깔끔하고 전문적인 이미지를 생성하세요.

중요: 문제 텍스트는 포함하지 마세요. 오직 풀이/해설만 표시하세요.

해설 내용:
{solution_text}

스타일 요구사항:
- 흰색 배경 (순수 흰색, #FFFFFF)
- 검은색 텍스트
- 명확하고 읽기 쉬운 글꼴
- 깔끔하고 심플한 교과서 스타일
- 수학적 정밀성
- 추가 장식이나 색상 없음
- 한국어 텍스트 사용
- LaTeX 수식은 올바르게 렌더링
- 단계별로 깔끔하게 정리"""

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Image']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                print(f"Generated solution image: {len(image_data)} bytes")
                return ensure_white_background(image_data)

        print("No image in Gemini response for solution")
        return None

    except Exception as e:
        print(f"Solution image generation failed: {e}")
        return None


def generate_twin_image_from_original(api_key, original_image_data, number_changes):
    """
    Generate a twin question image by editing numbers on the original image.
    Keeps the graph/diagram structure identical, only replaces specified numbers.

    Args:
        api_key: Gemini API key
        original_image_data: Original image data as bytes
        number_changes: List of number change descriptions, e.g. ["40° → 30°", "60° → 70°"]

    Returns:
        Image data as bytes, or None if generation fails
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        changes_text = "\n".join(f"- {change}" for change in number_changes)

        prompt = f"""이 수학 문제 이미지를 거의 동일하게 다시 그리되, 아래 숫자만 변경하세요.

변경할 숫자:
{changes_text}

★★★ 중요 규칙 ★★★
- 그래프, 도형, 그림의 형태와 구조는 원본과 완전히 동일하게 유지하세요
- 점의 위치, 선의 연결, 도형의 모양은 절대 바꾸지 마세요
- 변수명, 점 이름(A, B, C, D, O 등)은 원본 그대로 유지하세요
- 문제 텍스트의 구조도 원본과 동일하게 유지하고 숫자만 변경하세요
- 선택지가 있으면 선택지도 새 숫자에 맞게 변경하세요
- 흰색 배경 (#FFFFFF)
- 검은색 텍스트와 선
- 원본과 동일한 레이아웃과 스타일"""

        # Send original image with the editing prompt
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[
                prompt,
                types.Part.from_bytes(data=original_image_data, mime_type="image/png")
            ],
            config=types.GenerateContentConfig(
                response_modalities=['Image']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                print(f"Generated twin image from original: {len(image_data)} bytes")
                return ensure_white_background(image_data)

        print("No image in Gemini response for twin image editing")
        return None

    except Exception as e:
        print(f"Twin image editing failed: {e}")
        return None


def generate_question_image(api_key, latex_string, choices=None, graph_description=None, has_graph=False):
    """
    Generate a complete question image using Gemini's image generation model.

    The image includes:
    - Question text (latexString) at the top
    - Graph/diagram in the middle (if has_graph and graph_description provided)
    - Multiple choice options at the bottom (if choices provided)

    Args:
        api_key: Gemini API key
        latex_string: The question text in LaTeX format (displayed at top)
        choices: List of choice strings for MCQ (e.g., ["① 1", "② 2", ...])
        graph_description: Detailed description of the graph to generate (optional)
        has_graph: Whether the question has a graph/diagram

    Returns:
        Image data as bytes, or None if generation fails
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Build the prompt based on what content we have
        if has_graph and graph_description:
            # Question with graph and possibly choices
            choices_text = ""
            if choices and len(choices) > 0:
                choices_text = f"""

아래쪽에 다음 선택지들을 표시하세요:
{chr(10).join(choices)}"""

            prompt = f"""수학 교과서용 깔끔하고 전문적인 문제 이미지를 생성하세요.

이미지 구성:
1. 상단: 다음 문제 텍스트를 표시하세요 (LaTeX 수식 포함):
{latex_string}

2. 중간: 다음 그래프/도형을 그리세요:
{graph_description}
{choices_text}

스타일 요구사항:
- 흰색 배경
- 검은색 텍스트와 선
- 명확한 축 레이블 (x, y)
- 곡선 근처에 방정식 레이블
- 지정된 경우 색칠된 영역
- 깔끔하고 심플한 교과서 스타일
- 수학적 정밀성
- 추가 장식이나 색상 없음
- 한국어 텍스트 사용"""

        else:
            # Text-only question (no graph)
            if choices and len(choices) > 0:
                # MCQ without graph
                prompt = f"""수학 교과서용 깔끔하고 전문적인 문제 이미지를 생성하세요.

이미지 구성:
1. 상단: 다음 문제 텍스트를 표시하세요 (LaTeX 수식 포함):
{latex_string}

2. 하단: 다음 선택지들을 표시하세요:
{chr(10).join(choices)}

스타일 요구사항:
- 흰색 배경
- 검은색 텍스트
- 명확하고 읽기 쉬운 글꼴
- 깔끔하고 심플한 교과서 스타일
- 수학적 정밀성
- 추가 장식이나 색상 없음
- 한국어 텍스트 사용
- LaTeX 수식은 올바르게 렌더링"""
            else:
                # Text-only, no choices (short answer)
                prompt = f"""수학 교과서용 깔끔하고 전문적인 문제 이미지를 생성하세요.

이미지 구성:
문제 텍스트를 표시하세요 (LaTeX 수식 포함):
{latex_string}

스타일 요구사항:
- 흰색 배경
- 검은색 텍스트
- 명확하고 읽기 쉬운 글꼴
- 깔끔하고 심플한 교과서 스타일
- 수학적 정밀성
- 추가 장식이나 색상 없음
- 한국어 텍스트 사용
- LaTeX 수식은 올바르게 렌더링"""

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Image']
            )
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                print(f"Generated question image: {len(image_data)} bytes")
                return ensure_white_background(image_data)

        print("No image in Gemini response")
        return None

    except Exception as e:
        print(f"Question image generation failed: {e}")
        return None


def generate_new_graph_image(api_key, graph_description):
    """
    Generate a new graph image using Gemini's image generation model.
    (Legacy function - kept for backward compatibility)

    Args:
        api_key: Gemini API key
        graph_description: Detailed description of the graph to generate

    Returns:
        Image data as bytes, or None if generation fails
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = f"""Generate a clean, professional mathematical graph image for a textbook:

{graph_description}

Style requirements:
- White background
- Black lines for axes and curves
- Clear axis labels (x and y)
- Equations labeled near their curves
- Shaded regions if specified
- Simple, clean textbook style
- Mathematical precision
- No extra decorations or colors"""

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Image']
            )
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                print(f"Generated image: {len(image_data)} bytes")
                return image_data

        print("No image in Gemini response")
        return None

    except Exception as e:
        print(f"Image generation failed: {e}")
        return None


@app.route('/math_twin', methods=['POST'])
def generate_math_twin():
    """
    Generate a twin math question from an image using Gemini AI.
    If the image has a graph, generates a completely new graph image.

    Expects: multipart/form-data with 'image' file
    Returns: JSON with 'question', 'answer', 'solution' in LaTeX format, plus generated graph image
    """
    # Check if Gemini API key is configured
    if not app.config.get('GEMINI_API_KEY'):
        return jsonify({'error': 'Gemini API key not configured. Set GEMINI_API_KEY environment variable.'}), 500

    # Validate request has image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'error': 'Invalid image file type'}), 400

    # Read image data
    image_data = image_file.read()

    if len(image_data) > app.config['MAX_IMAGE_SIZE']:
        return jsonify({'error': 'Image file too large'}), 400

    # Get file extension for mime type
    filename = secure_filename(image_file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'png'
    mime_type = f"image/{ext}"
    if ext == 'jpg':
        mime_type = 'image/jpeg'

    try:
        # Configure Gemini API
        genai.configure(api_key=app.config['GEMINI_API_KEY'])

        # Create the model for analysis
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create the prompt - Korean version with choices support
        prompt = """이 수학 문제 이미지를 분석하고 "쌍둥이" 문제를 생성하세요.

★★★ 가장 중요한 규칙 ★★★
쌍둥이 문제는 원본 문제에서 "숫자(상수)"만 바꾼 문제입니다.
- 수식의 구조, 형태, 변수명은 절대 변경하지 마세요
- 변수명(x, y, z, a, b 등)은 원본 그대로 유지하세요
- 함수의 종류(sin, cos, log 등)는 원본 그대로 유지하세요
- 문장 구조와 질문 형태는 원본 그대로 유지하세요
- 오직 숫자(계수, 상수, 좌표값, 각도 등)만 다른 값으로 바꾸세요

예시:
- 원본: "∠BDC = 60°, ∠ABD = 40°일 때, ∠ACB의 크기는?"
- 쌍둥이: "∠BDC = 70°, ∠ABD = 30°일 때, ∠ACB의 크기는?"
  (문장 구조 동일, 60→70, 40→30만 변경)

- 원본: "함수 f(x) = x² - 3x + 2의 최솟값을 구하시오"
- 쌍둥이: "함수 f(x) = x² - 5x + 4의 최솟값을 구하시오"
  (f(x), x² 구조 그대로, 계수만 변경)

그래프/도형이 있는 경우:
- 그래프/도형의 구조는 원본과 완전히 동일하게 유지
- 이미지 안에 표시된 숫자(각도, 길이, 좌표 등)만 변경
- number_changes에 "원본숫자 → 새숫자" 형식으로 모든 변경사항을 기록하세요

중요: 먼저 이미지에 그래프, 도형, 그림이 포함되어 있는지 확인하세요.
또한 번호가 매겨진 보기가 있는 객관식 문제(MCQ)인지 확인하세요.

다음 JSON 형식으로만 응답하세요 (마크다운 없이, 코드 블록 없이, 순수 JSON만):

그래프/도형이 있는 경우:
{
    "question": "쌍둥이 수학 문제 (LaTeX 형식, 한국어로 작성)",
    "answer": "최종 답 (LaTeX 형식)",
    "solution": "단계별 풀이 (LaTeX 형식, 한국어로 작성)",
    "is_mcq": true,
    "choices": ["① 선택지1", "② 선택지2", "③ 선택지3", "④ 선택지4", "⑤ 선택지5"],
    "has_graph": true,
    "number_changes": ["40° → 30°", "60° → 70°"]
}

그래프가 없는 텍스트 전용 문제:
{
    "question": "쌍둥이 수학 문제 (LaTeX 형식, 한국어로 작성)",
    "answer": "최종 답 (LaTeX 형식)",
    "solution": "단계별 풀이 (LaTeX 형식, 한국어로 작성)",
    "is_mcq": true,
    "choices": ["① 선택지1", "② 선택지2", "③ 선택지3", "④ 선택지4", "⑤ 선택지5"],
    "has_graph": false,
    "number_changes": []
}

객관식이 아닌 경우:
{
    "question": "쌍둥이 수학 문제 (LaTeX 형식, 한국어로 작성)",
    "answer": "최종 답 (LaTeX 형식)",
    "solution": "단계별 풀이 (LaTeX 형식, 한국어로 작성)",
    "is_mcq": false,
    "choices": [],
    "has_graph": false,
    "number_changes": []
}

중요 지침:
- 모든 내용(문제, 풀이, 선택지)은 반드시 한국어로 작성하세요
- ★ 수식 구조/변수명/함수 종류는 절대 바꾸지 말고, 숫자(상수/계수/각도)만 변경하세요 ★
- is_mcq는 선택할 수 있는 번호가 매겨진 보기가 있으면 true
- choices는 객관식인 경우 모든 보기를 배열로 제공 (①②③④⑤ 기호 포함)
- 객관식이 아니면 choices는 빈 배열 []
- number_changes는 이미지 안의 숫자를 어떻게 바꿀지 "원본 → 변경" 형식의 배열 (그래프/도형이 있을 때만)
- LaTeX 형식으로 모든 수학 표현식 작성
- 풀 수 있는 문제로 만들고 명확한 답을 포함하세요
- 같은 난이도를 유지하세요
- 유효한 JSON만 응답하세요, 추가 텍스트 없이"""

        # Prepare the image for Gemini
        image_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode('utf-8')
        }

        # Generate response
        response = model.generate_content([prompt, image_part])

        # Parse the response
        response_text = extract_json_from_response(response.text)

        # Parse JSON response
        result = json.loads(response_text)

        # Validate required fields
        if 'question' not in result or 'answer' not in result or 'solution' not in result:
            return jsonify({
                'error': 'Invalid response from Gemini API',
                'raw_response': response.text
            }), 500

        # Extract question data
        latex_string = result.get('question', '')
        choices = result.get('choices', [])
        has_graph = result.get('has_graph', False)
        number_changes = result.get('number_changes', [])
        is_mcq = result.get('is_mcq', False)

        # Generate question image
        generated_image_uuid = None
        try:
            if has_graph and number_changes:
                # Graph/diagram exists: edit original image, only change numbers
                print(f"Editing original image (changing numbers: {number_changes})...")
                generated_image_data = generate_twin_image_from_original(
                    api_key=app.config['GEMINI_API_KEY'],
                    original_image_data=image_data,
                    number_changes=number_changes
                )
            else:
                # No graph: generate new text-only image
                print(f"Generating question image (is_mcq={is_mcq}, choices={len(choices)})...")
                generated_image_data = generate_question_image(
                    api_key=app.config['GEMINI_API_KEY'],
                    latex_string=latex_string,
                    choices=choices if is_mcq else None,
                    graph_description=None,
                    has_graph=False
                )

            if generated_image_data:
                # Save generated image to disk
                generated_image_uuid = str(uuid.uuid4())
                image_path = os.path.join(app.config['IMAGE_FOLDER_TWIN'], f"{generated_image_uuid}.png")
                with open(image_path, 'wb') as f:
                    f.write(generated_image_data)
                print(f"Saved generated question image: {generated_image_uuid}")
            else:
                print("Question image generation returned no data")

        except Exception as e:
            # If image generation fails, continue without it
            print(f"Warning: Question image generation failed: {e}")

        # Generate a separate solution-only image (no question text)
        solution_image_uuid = None
        try:
            solution_text = result.get('solution', '')
            if solution_text:
                print("Generating solution-only image...")
                solution_image_data = generate_solution_image(
                    api_key=app.config['GEMINI_API_KEY'],
                    solution_text=solution_text
                )
                if solution_image_data:
                    solution_image_uuid = str(uuid.uuid4())
                    solution_image_path = os.path.join(app.config['IMAGE_FOLDER_TWIN'], f"{solution_image_uuid}.png")
                    with open(solution_image_path, 'wb') as f:
                        f.write(solution_image_data)
                    print(f"Saved solution image: {solution_image_uuid}")
                else:
                    print("Solution image generation returned no data")
        except Exception as e:
            print(f"Warning: Solution image generation failed: {e}")

        response_data = {
            'question': latex_string,
            'answer': result['answer'],
            'solution': result['solution'],
            'is_mcq': is_mcq,
            'choices': choices,
            'has_graph': has_graph,
            'number_changes': number_changes if has_graph else None,
            'modified_image_id': generated_image_uuid,
            'modified_image_url': f"/math_twin/images/{generated_image_uuid}" if generated_image_uuid else None,
            'solution_image_id': solution_image_uuid,
            'solution_image_url': f"/math_twin/images/{solution_image_uuid}" if solution_image_uuid else None
        }

        return jsonify(response_data)

    except json.JSONDecodeError as e:
        return jsonify({
            'error': 'Failed to parse Gemini response as JSON',
            'details': str(e),
            'raw_response': response.text if 'response' in locals() else None
        }), 500
    except Exception as e:
        return jsonify({
            'error': 'Failed to generate twin question',
            'details': str(e)
        }), 500


def download_image_from_url(url):
    """
    Download image from URL and return image data as bytes.
    """
    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return None


def generate_single_twin(api_key, image_data, original_url, base_url):
    """
    Generate a single twin question from image data.
    Returns formatted response dict or None if failed.
    """
    try:
        # Determine mime type from image data
        img = Image.open(io.BytesIO(image_data))
        mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
        if mime_type == "image/jpeg":
            mime_type = "image/jpeg"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """이 수학 문제 이미지를 분석하고 "쌍둥이" 문제를 생성하세요.

★★★ 가장 중요한 규칙 ★★★
쌍둥이 문제는 원본 문제에서 "숫자(상수)"만 바꾼 문제입니다.
- 수식의 구조, 형태, 변수명은 절대 변경하지 마세요
- 변수명(x, y, z, a, b 등)은 원본 그대로 유지하세요
- 함수의 종류(sin, cos, log 등)는 원본 그대로 유지하세요
- 문장 구조와 질문 형태는 원본 그대로 유지하세요
- 오직 숫자(계수, 상수, 좌표값, 각도 등)만 다른 값으로 바꾸세요

그래프/도형이 있는 경우:
- 그래프/도형의 구조는 원본과 완전히 동일하게 유지
- 이미지 안에 표시된 숫자(각도, 길이, 좌표 등)만 변경
- number_changes에 "원본숫자 → 새숫자" 형식으로 모든 변경사항을 기록하세요

중요: 먼저 이미지에 그래프, 도형, 그림이 포함되어 있는지 확인하세요.
또한 번호가 매겨진 보기가 있는 객관식 문제(MCQ)인지 확인하세요.

다음 JSON 형식으로만 응답하세요 (마크다운 없이, 코드 블록 없이, 순수 JSON만):

그래프/도형이 있는 경우:
{
    "question": "쌍둥이 수학 문제 (LaTeX 형식, 전체 문제 텍스트, 한국어로 작성)",
    "answer_number": 1,
    "solution": "단계별 풀이/설명 (LaTeX 형식, 한국어로 작성)",
    "is_mcq": true,
    "choices": ["① 선택지1", "② 선택지2", "③ 선택지3", "④ 선택지4", "⑤ 선택지5"],
    "has_graph": true,
    "number_changes": ["40° → 30°", "60° → 70°"]
}

그래프가 없는 텍스트 전용 문제:
{
    "question": "쌍둥이 수학 문제 (LaTeX 형식, 한국어로 작성)",
    "answer_number": 3,
    "solution": "단계별 풀이 (LaTeX 형식, 한국어로 작성)",
    "is_mcq": true,
    "choices": ["① 선택지1", "② 선택지2", "③ 선택지3", "④ 선택지4", "⑤ 선택지5"],
    "has_graph": false,
    "number_changes": []
}

객관식이 아닌 경우:
{
    "question": "쌍둥이 수학 문제 (LaTeX 형식, 한국어로 작성)",
    "answer_number": 0,
    "solution": "단계별 풀이 (LaTeX 형식, 한국어로 작성)",
    "is_mcq": false,
    "choices": [],
    "has_graph": false,
    "number_changes": []
}

중요 지침:
- 모든 내용(문제, 풀이, 선택지)은 반드시 한국어로 작성하세요
- ★ 수식 구조/변수명/함수 종류는 절대 바꾸지 말고, 숫자(상수/계수/각도)만 변경하세요 ★
- answer_number는 객관식 문제의 정답 번호(1, 2, 3, 4, 또는 5)
- is_mcq는 선택할 수 있는 번호가 매겨진 보기가 있으면 true
- choices는 객관식인 경우 모든 보기를 배열로 제공 (①②③④⑤ 기호 포함)
- 객관식이 아니면 choices는 빈 배열 []
- number_changes는 이미지 안의 숫자를 어떻게 바꿀지 "원본 → 변경" 형식의 배열 (그래프/도형이 있을 때만)
- 유효한 JSON만 응답하세요, 추가 텍스트 없이"""

        image_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode('utf-8')
        }

        response = model.generate_content([prompt, image_part])
        response_text = extract_json_from_response(response.text)
        result = json.loads(response_text)

        # Extract question data
        latex_string = result.get('question', '')
        choices = result.get('choices', [])
        has_graph = result.get('has_graph', False)
        number_changes = result.get('number_changes', [])
        is_mcq = result.get('is_mcq', True)

        # Generate question image
        generated_image_url = None
        try:
            if has_graph and number_changes:
                # Graph/diagram exists: edit original image, only change numbers
                print(f"Editing original image (changing numbers: {number_changes})...")
                generated_image_data = generate_twin_image_from_original(
                    api_key=api_key,
                    original_image_data=image_data,
                    number_changes=number_changes
                )
            else:
                # No graph: generate new text-only image
                print(f"Generating question image (is_mcq={is_mcq}, choices={len(choices)})...")
                generated_image_data = generate_question_image(
                    api_key=api_key,
                    latex_string=latex_string,
                    choices=choices if is_mcq else None,
                    graph_description=None,
                    has_graph=False
                )

            if generated_image_data:
                generated_image_uuid = str(uuid.uuid4())
                image_path = os.path.join(app.config['IMAGE_FOLDER_TWIN'], f"{generated_image_uuid}.png")
                with open(image_path, 'wb') as f:
                    f.write(generated_image_data)
                generated_image_url = f"{base_url}/math_twin/images/{generated_image_uuid}"
                print(f"Saved question image: {generated_image_uuid}")
            else:
                print("Question image generation returned no data")
        except Exception as e:
            print(f"Warning: Question image generation failed: {e}")

        # Generate a separate solution-only image (no question text)
        solution_image_url = None
        try:
            solution_text = result.get('solution', '')
            if solution_text:
                print("Generating solution-only image...")
                solution_image_data = generate_solution_image(
                    api_key=api_key,
                    solution_text=solution_text
                )
                if solution_image_data:
                    solution_image_uuid = str(uuid.uuid4())
                    solution_image_path = os.path.join(app.config['IMAGE_FOLDER_TWIN'], f"{solution_image_uuid}.png")
                    with open(solution_image_path, 'wb') as f:
                        f.write(solution_image_data)
                    solution_image_url = f"{base_url}/math_twin/images/{solution_image_uuid}"
                    print(f"Saved solution image: {solution_image_uuid}")
                else:
                    print("Solution image generation returned no data")
        except Exception as e:
            print(f"Warning: Solution image generation failed: {e}")

        return {
            "latexString": latex_string,
            "answerString": result.get('solution', ''),
            "originalImageURL": original_url,
            "questionImageUrl": generated_image_url,
            "answerImageUrl": solution_image_url if solution_image_url else generated_image_url,
            "isMCQ": is_mcq,
            "choices": choices,
            "answer": result.get('answer_number', 1)
        }

    except Exception as e:
        print(f"Error generating twin: {e}")
        return None


@app.route('/math_twin/batch', methods=['POST'])
def generate_math_twin_batch():
    """
    Generate multiple twin questions from a list of image URLs in parallel.

    Expects JSON body:
    {
        "questions": ["https://example.com/img1.png", "https://example.com/img2.png"],
        "num_questions": 4,
        "max_workers": 10  // optional, default 10
    }

    Returns: List of generated twin questions
    """
    # Check if Gemini API key is configured
    if not app.config.get('GEMINI_API_KEY'):
        return jsonify({'error': 'Gemini API key not configured'}), 500

    # Parse JSON body
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    questions = data.get('questions', [])
    num_questions = data.get('num_questions', 1)
    max_workers = data.get('max_workers', 10)  # Default 10 parallel workers

    if not questions:
        return jsonify({'error': 'No questions (image URLs) provided'}), 400

    if not isinstance(questions, list):
        return jsonify({'error': 'questions must be a list of URLs'}), 400

    if not isinstance(num_questions, int) or num_questions < 1:
        return jsonify({'error': 'num_questions must be a positive integer'}), 400

    # Get base URL for generated images
    base_url = request.host_url.rstrip('/')
    api_key = app.config['GEMINI_API_KEY']

    # First, download all images in parallel
    print(f"Downloading {len(questions)} images...")
    image_data_map = {}

    def download_task(url):
        return url, download_image_from_url(url)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        download_futures = {executor.submit(download_task, url): url for url in questions}
        for future in as_completed(download_futures):
            url, image_data = future.result()
            if image_data:
                image_data_map[url] = image_data
                print(f"  Downloaded: {url}")
            else:
                print(f"  Failed to download: {url}")

    # Prepare all tasks for twin generation
    tasks = []
    for image_url, image_data in image_data_map.items():
        for i in range(num_questions):
            tasks.append((image_url, image_data, i))

    print(f"Generating {len(tasks)} twins in parallel (max_workers={max_workers})...")

    # Generate all twins in parallel
    results = []

    def generate_task(task_info):
        image_url, image_data, task_index = task_info
        print(f"  Generating twin {task_index+1} for {image_url}")
        return generate_single_twin(api_key, image_data, image_url, base_url)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_task, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    task = futures[future]
                    print(f"  Failed to generate twin for {task[0]}")
            except Exception as e:
                print(f"  Error in parallel task: {e}")

    print(f"Generated {len(results)} twins successfully")
    return jsonify(results)


if __name__ == '__main__':
    import argparse
    import ssl

    parser = argparse.ArgumentParser(description='Math Problems Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--https', action='store_true', help='Enable HTTPS')
    parser.add_argument('--cert', default='certs/cert.pem', help='SSL certificate file (default: certs/cert.pem)')
    parser.add_argument('--key', default='certs/key.pem', help='SSL private key file (default: certs/key.pem)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    if args.https:
        # Check if certificate files exist
        if not os.path.exists(args.cert) or not os.path.exists(args.key):
            print(f"Error: SSL certificate files not found!")
            print(f"  Certificate: {args.cert}")
            print(f"  Private key: {args.key}")
            print("\nGenerate self-signed certificates with:")
            print("  ./generate_certs.sh")
            print("\nOr for production, use Let's Encrypt:")
            print("  sudo certbot certonly --standalone -d yourdomain.com")
            exit(1)

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert, args.key)
        print(f"Starting HTTPS server on https://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug, ssl_context=ssl_context)
    else:
        print(f"Starting HTTP server on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
