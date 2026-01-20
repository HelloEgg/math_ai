# Math Problems Server

A Flask-based REST API server for storing, searching, and retrieving math problems with solutions and audio explanations.

## Features

- **Image-based search**: Find existing math problems by uploading an image
- **Store math problems**: Save problems with images, LaTeX solutions, and audio explanations
- **Retrieve solutions**: Get solutions and audio explanations by UUID

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd math_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will start on `http://localhost:5000`.

## API Endpoints

### Health Check

```
GET /health
```

Returns server status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Math Problems Server is running"
}
```

---

### Search for Math Problem

```
POST /search
```

Search for an existing math problem by image.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| image | File | Math problem image (png, jpg, jpeg, gif, webp) |

**Response:**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

Or if not found:
```json
{
  "uuid": null
}
```

---

### Create Math Problem

```
POST /problems
```

Create a new math problem with solution and audio explanation.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| image | File | Math problem image (png, jpg, jpeg, gif, webp) |
| solution_latex | String | Solution in LaTeX format |
| audio | File | Audio explanation (mp3, wav, ogg, m4a, webm) |

**Response (201):**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Math problem created successfully"
}
```

**Response (409 - Duplicate):**
```json
{
  "error": "A problem with this exact image already exists",
  "existing_uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### Get Math Problem

```
GET /problems/<uuid>
```

Retrieve a math problem's solution and metadata.

**Response:**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "solution_latex": "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}",
  "audio_url": "/problems/550e8400-e29b-41d4-a716-446655440000/audio",
  "image_url": "/problems/550e8400-e29b-41d4-a716-446655440000/image",
  "created_at": "2024-01-15T10:30:00"
}
```

---

### Get Audio File

```
GET /problems/<uuid>/audio
```

Download the audio explanation file.

**Response:** Audio file (binary)

---

### Get Image File

```
GET /problems/<uuid>/image
```

Download the math problem image.

**Response:** Image file (binary)

---

### List All Problems

```
GET /problems
```

List all math problems with pagination.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page (max 100) |

**Response:**
```json
{
  "problems": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "image_url": "/problems/550e8400-e29b-41d4-a716-446655440000/image",
      "created_at": "2024-01-15T10:30:00"
    }
  ],
  "total": 50,
  "page": 1,
  "per_page": 20,
  "pages": 3
}
```

---

### Delete Math Problem

```
DELETE /problems/<uuid>
```

Delete a math problem and its associated files.

**Response:**
```json
{
  "message": "Problem deleted successfully"
}
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| SECRET_KEY | Flask secret key | dev-secret-key-change-in-production |
| DATABASE_URL | Database connection URL | sqlite:///math_problems.db |

## File Limits

- **Images:** Max 10 MB (png, jpg, jpeg, gif, webp)
- **Audio:** Max 50 MB (mp3, wav, ogg, m4a, webm)

## Project Structure

```
math_ai/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── models.py           # Database models
├── requirements.txt    # Python dependencies
├── uploads/            # Uploaded files (auto-created)
│   ├── images/         # Math problem images
│   └── audio/          # Audio explanations
└── README.md           # This file
```

## License

MIT
