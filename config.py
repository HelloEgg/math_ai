import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Application configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "math_problems.db")}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File uploads - General
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
    AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')

    # File uploads - Summary
    IMAGE_FOLDER_SUMMARY = os.path.join(UPLOAD_FOLDER, 'images_summary')
    AUDIO_FOLDER_SUMMARY = os.path.join(UPLOAD_FOLDER, 'audio_summary')

    # File uploads - Deep
    IMAGE_FOLDER_DEEP = os.path.join(UPLOAD_FOLDER, 'images_deep')
    AUDIO_FOLDER_DEEP = os.path.join(UPLOAD_FOLDER, 'audio_deep')

    # File uploads - Original
    IMAGE_FOLDER_ORIGINAL = os.path.join(UPLOAD_FOLDER, 'images_original')

    # File uploads - English
    IMAGE_FOLDER_ENGLISH = os.path.join(UPLOAD_FOLDER, 'images_english')
    IMAGE_FOLDER_ENGLISH_SUMMARY = os.path.join(UPLOAD_FOLDER, 'images_english_summary')
    IMAGE_FOLDER_ENGLISH_DEEP = os.path.join(UPLOAD_FOLDER, 'images_english_deep')
    AUDIO_FOLDER_ENGLISH = os.path.join(UPLOAD_FOLDER, 'audio_english')
    AUDIO_FOLDER_ENGLISH_SUMMARY = os.path.join(UPLOAD_FOLDER, 'audio_english_summary')
    AUDIO_FOLDER_ENGLISH_DEEP = os.path.join(UPLOAD_FOLDER, 'audio_english_deep')

    # File uploads - Science
    IMAGE_FOLDER_SCIENCE = os.path.join(UPLOAD_FOLDER, 'images_science')
    IMAGE_FOLDER_SCIENCE_SUMMARY = os.path.join(UPLOAD_FOLDER, 'images_science_summary')
    IMAGE_FOLDER_SCIENCE_DEEP = os.path.join(UPLOAD_FOLDER, 'images_science_deep')
    AUDIO_FOLDER_SCIENCE = os.path.join(UPLOAD_FOLDER, 'audio_science')
    AUDIO_FOLDER_SCIENCE_SUMMARY = os.path.join(UPLOAD_FOLDER, 'audio_science_summary')
    AUDIO_FOLDER_SCIENCE_DEEP = os.path.join(UPLOAD_FOLDER, 'audio_science_deep')

    # File uploads - Social Science
    IMAGE_FOLDER_SOCIAL_SCIENCE = os.path.join(UPLOAD_FOLDER, 'images_social_science')
    IMAGE_FOLDER_SOCIAL_SCIENCE_SUMMARY = os.path.join(UPLOAD_FOLDER, 'images_social_science_summary')
    IMAGE_FOLDER_SOCIAL_SCIENCE_DEEP = os.path.join(UPLOAD_FOLDER, 'images_social_science_deep')
    AUDIO_FOLDER_SOCIAL_SCIENCE = os.path.join(UPLOAD_FOLDER, 'audio_social_science')
    AUDIO_FOLDER_SOCIAL_SCIENCE_SUMMARY = os.path.join(UPLOAD_FOLDER, 'audio_social_science_summary')
    AUDIO_FOLDER_SOCIAL_SCIENCE_DEEP = os.path.join(UPLOAD_FOLDER, 'audio_social_science_deep')

    # File uploads - Korean
    IMAGE_FOLDER_KOREAN = os.path.join(UPLOAD_FOLDER, 'images_korean')
    IMAGE_FOLDER_KOREAN_SUMMARY = os.path.join(UPLOAD_FOLDER, 'images_korean_summary')
    IMAGE_FOLDER_KOREAN_DEEP = os.path.join(UPLOAD_FOLDER, 'images_korean_deep')
    AUDIO_FOLDER_KOREAN = os.path.join(UPLOAD_FOLDER, 'audio_korean')
    AUDIO_FOLDER_KOREAN_SUMMARY = os.path.join(UPLOAD_FOLDER, 'audio_korean_summary')
    AUDIO_FOLDER_KOREAN_DEEP = os.path.join(UPLOAD_FOLDER, 'audio_korean_deep')

    # File uploads - Math Twin
    IMAGE_FOLDER_TWIN = os.path.join(UPLOAD_FOLDER, 'images_twin')

    # Allowed extensions
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'webm'}

    # Max file sizes (in bytes)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB

    # Gemini API
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
