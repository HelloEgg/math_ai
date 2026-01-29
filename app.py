import os
import hashlib
import json
import base64
import io
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import pytesseract

from config import Config
from models import db, MathProblem, MathProblemSummary, MathProblemDeep


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
    # Create upload directories - Math Twin
    os.makedirs(app.config['IMAGE_FOLDER_TWIN'], exist_ok=True)

    # Create database tables
    with app.app_context():
        db.create_all()

    return app


app = create_app()


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def compute_image_hash(file_data):
    """Compute SHA-256 hash of image data for exact matching."""
    return hashlib.sha256(file_data).hexdigest()


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
    Search for a math problem by image.

    Expects: multipart/form-data with 'image' file
    Returns: JSON with 'uuid' (string or null)
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

    # Search for existing problem with same image hash
    problem = MathProblem.query.filter_by(image_hash=image_hash).first()

    if problem:
        return jsonify({'uuid': problem.id})
    else:
        return jsonify({'uuid': None})


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

    # Read audio data to check size
    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    # Create new problem entry
    problem = MathProblem(
        image_hash=image_hash,
        solution_latex=solution_latex,
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
# Summary Endpoints
# =============================================================================

@app.route('/search_summary', methods=['POST'])
def search_problem_summary():
    """Search for a math problem summary by image."""
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
    problem = MathProblemSummary.query.filter_by(image_hash=image_hash).first()

    if problem:
        return jsonify({'uuid': problem.id})
    else:
        return jsonify({'uuid': None})


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

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    problem = MathProblemSummary(
        image_hash=image_hash,
        solution_latex=solution_latex,
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
    """Search for a deep math problem by image."""
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
    problem = MathProblemDeep.query.filter_by(image_hash=image_hash).first()

    if problem:
        return jsonify({'uuid': problem.id})
    else:
        return jsonify({'uuid': None})


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

    audio_data = audio_file.read()
    if len(audio_data) > app.config['MAX_AUDIO_SIZE']:
        return jsonify({'error': 'Audio file too large'}), 400

    problem = MathProblemDeep(
        image_hash=image_hash,
        solution_latex=solution_latex,
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


def find_text_locations_ocr(img):
    """
    Use OCR to find all text locations in an image.
    Returns a list of dicts with text, x, y, width, height.
    """
    # Get OCR data with bounding boxes
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    text_locations = []
    n_boxes = len(ocr_data['text'])

    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])

        # Skip empty text or low confidence
        if text and conf > 0:
            text_locations.append({
                'text': text,
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': conf
            })

    return text_locations


def modify_image_with_replacements(image_data, replacements):
    """
    Modify image by using OCR to find text locations and replacing them.

    replacements: list of dicts with keys:
        - old_text: original text to find and cover
        - new_text: new text to write
    """
    # Open image
    img = Image.open(io.BytesIO(image_data))

    # Convert to RGB if necessary (handles PNG with transparency)
    if img.mode in ('RGBA', 'P'):
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Find all text locations using OCR
    text_locations = find_text_locations_ocr(img)

    draw = ImageDraw.Draw(img)

    # Track which replacements have been made to avoid duplicates
    replacements_made = []

    for replacement in replacements:
        old_text = replacement.get('old_text', '').strip()
        new_text = replacement.get('new_text', '').strip()

        if not old_text:
            continue

        # Find all matching text locations
        for loc in text_locations:
            # Check if this location matches and hasn't been replaced yet
            loc_key = (loc['x'], loc['y'])
            if loc['text'] == old_text and loc_key not in replacements_made:
                try:
                    x, y = loc['x'], loc['y']
                    width, height = loc['width'], loc['height']

                    # Load font with appropriate size based on detected text height
                    font = None
                    font_size = max(height - 2, 10)  # Slightly smaller than box height
                    font_paths = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                        "/usr/share/fonts/TTF/DejaVuSans.ttf",
                        "/System/Library/Fonts/Helvetica.ttc",
                        "C:\\Windows\\Fonts\\arial.ttf",
                    ]
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            try:
                                font = ImageFont.truetype(font_path, font_size)
                                break
                            except:
                                continue

                    if font is None:
                        font = ImageFont.load_default()

                    # Draw white rectangle to cover old text
                    padding = 2
                    draw.rectangle(
                        [x - padding, y - padding, x + width + padding, y + height + padding],
                        fill='white'
                    )

                    # Draw new text centered in the box
                    # Get text bounding box for centering
                    text_bbox = draw.textbbox((0, 0), new_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # Center the text in the original box
                    text_x = x + (width - text_width) // 2
                    text_y = y + (height - text_height) // 2

                    draw.text((text_x, text_y), new_text, fill='black', font=font)

                    replacements_made.append(loc_key)
                    print(f"Replaced '{old_text}' with '{new_text}' at ({x}, {y})")

                except Exception as e:
                    print(f"Warning: Failed to apply replacement for '{old_text}': {e}")
                    continue

    # Save to bytes
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    return output.getvalue()


@app.route('/math_twin', methods=['POST'])
def generate_math_twin():
    """
    Generate a twin math question from an image using Gemini AI.
    Also modifies the image to show the new numbers.

    Expects: multipart/form-data with 'image' file
    Returns: JSON with 'question', 'answer', 'solution' in LaTeX format, plus 'modified_image' as base64
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

        # Create the model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create the prompt - OCR will find text locations automatically
        prompt = """Analyze this math problem image and create a "twin" problem.

A twin problem has the SAME structure and type of question, but with DIFFERENT numbers and/or variable names.

For example:
- Original: "What is y when y = 2x and x = 3?"
- Twin: "What is z when z = 5w and w = 4?"

IMPORTANT: First determine if the image contains any graphs, diagrams, figures, or geometric shapes.
- If YES (contains graphs/diagrams/figures): Provide replacements specifying which text/numbers to change
- If NO (text-only problem): Set "has_graph" to false and leave "replacements" as an empty array

Please respond in the following JSON format ONLY (no markdown, no code blocks, just pure JSON):
{
    "question": "The twin math question in LaTeX format",
    "answer": "The final answer in LaTeX format",
    "solution": "Step-by-step solution in LaTeX format",
    "has_graph": true,
    "replacements": [
        {"old_text": "2", "new_text": "3"},
        {"old_text": "5", "new_text": "6"}
    ]
}

For text-only problems (no graphs):
{
    "question": "The twin math question in LaTeX format",
    "answer": "The final answer in LaTeX format",
    "solution": "Step-by-step solution in LaTeX format",
    "has_graph": false,
    "replacements": []
}

Important:
- Use LaTeX formatting for all mathematical expressions in question/answer/solution
- The twin question should be solvable and have a clear answer
- Make sure numbers are different from the original
- Keep the same difficulty level
- ONLY provide replacements if the image contains graphs, diagrams, or figures
- For text-only math problems, do NOT provide replacements
- In replacements, only specify old_text and new_text (coordinates will be detected automatically via OCR)
- Respond with valid JSON only, no additional text"""

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

        # Modify the image only if has_graph is true and replacements are provided
        modified_image_uuid = None
        has_graph = result.get('has_graph', False)
        if has_graph and 'replacements' in result and len(result['replacements']) > 0:
            try:
                modified_image_data = modify_image_with_replacements(image_data, result['replacements'])
                # Generate UUID and save image to disk
                modified_image_uuid = str(uuid.uuid4())
                image_path = os.path.join(app.config['IMAGE_FOLDER_TWIN'], f"{modified_image_uuid}.png")
                with open(image_path, 'wb') as f:
                    f.write(modified_image_data)
            except Exception as e:
                # If image modification fails, continue without it
                print(f"Warning: Image modification failed: {e}")

        response_data = {
            'question': result['question'],
            'answer': result['answer'],
            'solution': result['solution'],
            'has_graph': has_graph,
            'modified_image_id': modified_image_uuid,
            'modified_image_url': f"/math_twin/images/{modified_image_uuid}" if modified_image_uuid else None
        }

        if 'replacements' in result:
            response_data['replacements'] = result['replacements']

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
