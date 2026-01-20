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

## Running with HTTPS

### Option 1: Self-signed certificates (for development/testing)

```bash
# Generate self-signed certificates
./generate_certs.sh

# Start server with HTTPS
python app.py --https

# Or with custom port (443 requires sudo)
sudo python app.py --https --port 443
```

> **Note:** Self-signed certificates will show browser security warnings. Click "Advanced" → "Proceed" to continue.

### Option 2: Let's Encrypt (for production)

```bash
# Install certbot
sudo apt install certbot  # Ubuntu/Debian
# or
sudo yum install certbot  # CentOS/RHEL

# Generate certificate (stop any server on port 80 first)
sudo certbot certonly --standalone -d yourdomain.com

# Start server with Let's Encrypt certificates
sudo python app.py --https \
  --cert /etc/letsencrypt/live/yourdomain.com/fullchain.pem \
  --key /etc/letsencrypt/live/yourdomain.com/privkey.pem \
  --port 443
```

### Option 3: Reverse proxy with nginx (recommended for production)

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 50M;
    }
}
```

### Command-line options

```bash
python app.py --help

Options:
  --host HOST     Host to bind to (default: 0.0.0.0)
  --port PORT     Port to bind to (default: 5000)
  --https         Enable HTTPS
  --cert FILE     SSL certificate file (default: certs/cert.pem)
  --key FILE      SSL private key file (default: certs/key.pem)
  --debug         Enable debug mode
```

## Quick Start Examples

### 1. Create a new math problem
```bash
curl -X POST http://localhost:5000/problems \
  -F "image=@/path/to/math_problem.png" \
  -F "solution_latex=x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}" \
  -F "audio=@/path/to/explanation.mp3"
```

### 2. Search for existing problem by image
```bash
curl -X POST http://localhost:5000/search \
  -F "image=@/path/to/math_problem.png"
```

### 3. Get solution by UUID
```bash
curl http://localhost:5000/problems/550e8400-e29b-41d4-a716-446655440000
```

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

Search for an existing math problem by image. Uses SHA-256 hash for exact image matching.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | File | Yes | Math problem image (png, jpg, jpeg, gif, webp) |

**curl Example:**
```bash
# Search for a math problem by uploading an image
curl -X POST http://localhost:5000/search \
  -F "image=@./my_math_problem.png"

# Using full path
curl -X POST http://localhost:5000/search \
  -F "image=@/home/user/images/quadratic_equation.jpg"
```

**Response (Found):**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response (Not Found):**
```json
{
  "uuid": null
}
```

**Error Responses:**
```json
{"error": "No image file provided"}          // 400 - missing image field
{"error": "No image file selected"}          // 400 - empty file
{"error": "Invalid image file type"}         // 400 - unsupported format
{"error": "Image file too large"}            // 400 - exceeds 10MB
```

---

### Create Math Problem

```
POST /problems
```

Create a new math problem with solution and audio explanation.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | File | Yes | Math problem image (png, jpg, jpeg, gif, webp) - max 10MB |
| solution_latex | String | Yes | Solution in LaTeX format (text field) |
| audio | File | Yes | Audio explanation (mp3, wav, ogg, m4a, webm) - max 50MB |

**curl Examples:**
```bash
# Basic example
curl -X POST http://localhost:5000/problems \
  -F "image=@./problem.png" \
  -F "solution_latex=x = 5" \
  -F "audio=@./explanation.mp3"

# With complex LaTeX (use quotes for special characters)
curl -X POST http://localhost:5000/problems \
  -F "image=@./quadratic.jpg" \
  -F "solution_latex=x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}" \
  -F "audio=@./quadratic_explanation.wav"

# Full example with absolute paths
curl -X POST http://localhost:5000/problems \
  -F "image=@/home/user/math/integral_problem.png" \
  -F "solution_latex=\int_{0}^{1} x^2 dx = \frac{1}{3}" \
  -F "audio=@/home/user/audio/integral_explanation.mp3"
```

**JavaScript (fetch) Example:**
```javascript
const formData = new FormData();
formData.append('image', imageFile);                    // File object
formData.append('solution_latex', '\\frac{1}{2}');      // String
formData.append('audio', audioFile);                    // File object

const response = await fetch('http://localhost:5000/problems', {
  method: 'POST',
  body: formData
});
const result = await response.json();
```

**Python (requests) Example:**
```python
import requests

files = {
    'image': ('problem.png', open('problem.png', 'rb'), 'image/png'),
    'audio': ('explanation.mp3', open('explanation.mp3', 'rb'), 'audio/mpeg')
}
data = {
    'solution_latex': r'x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}'
}

response = requests.post('http://localhost:5000/problems', files=files, data=data)
print(response.json())
```

**Response (201 Created):**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Math problem created successfully"
}
```

**Response (409 Conflict - Duplicate Image):**
```json
{
  "error": "A problem with this exact image already exists",
  "existing_uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses (400 Bad Request):**
```json
{"error": "No image file provided"}
{"error": "No audio file provided"}
{"error": "No solution_latex provided"}
{"error": "solution_latex cannot be empty"}
{"error": "Invalid image file type"}
{"error": "Invalid audio file type"}
{"error": "Image file too large"}
{"error": "Audio file too large"}
```

---

### Get Math Problem

```
GET /problems/<uuid>
```

Retrieve a math problem's solution and metadata by UUID.

**URL Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| uuid | String | The UUID of the math problem |

**curl Examples:**
```bash
# Get problem details
curl http://localhost:5000/problems/550e8400-e29b-41d4-a716-446655440000

# Pretty print with jq
curl -s http://localhost:5000/problems/550e8400-e29b-41d4-a716-446655440000 | jq
```

**JavaScript Example:**
```javascript
const uuid = '550e8400-e29b-41d4-a716-446655440000';
const response = await fetch(`http://localhost:5000/problems/${uuid}`);
const problem = await response.json();

console.log(problem.solution_latex);  // LaTeX solution
console.log(problem.audio_url);       // URL to download audio
```

**Response (200 OK):**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "solution_latex": "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}",
  "audio_url": "/problems/550e8400-e29b-41d4-a716-446655440000/audio",
  "image_url": "/problems/550e8400-e29b-41d4-a716-446655440000/image",
  "created_at": "2024-01-15T10:30:00"
}
```

**Response (404 Not Found):**
```json
{"error": "Problem not found"}
```

---

### Get Audio File

```
GET /problems/<uuid>/audio
```

Download the audio explanation file for a math problem.

**curl Examples:**
```bash
# Download audio file
curl http://localhost:5000/problems/550e8400-e29b-41d4-a716-446655440000/audio \
  --output explanation.mp3

# Stream in browser or audio player
# Just open: http://localhost:5000/problems/<uuid>/audio
```

**Response:** Audio file (binary stream with appropriate Content-Type)

---

### Get Image File

```
GET /problems/<uuid>/image
```

Download the math problem image.

**curl Examples:**
```bash
# Download image file
curl http://localhost:5000/problems/550e8400-e29b-41d4-a716-446655440000/image \
  --output problem.png

# View in browser
# Just open: http://localhost:5000/problems/<uuid>/image
```

**Response:** Image file (binary stream with appropriate Content-Type)

---

### List All Problems

```
GET /problems
```

List all math problems with pagination.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number (1-indexed) |
| per_page | int | 20 | Items per page (max 100) |

**curl Examples:**
```bash
# Get first page (default 20 items)
curl http://localhost:5000/problems

# Get specific page with custom page size
curl "http://localhost:5000/problems?page=2&per_page=10"

# Get maximum items per page
curl "http://localhost:5000/problems?per_page=100"
```

**Response (200 OK):**
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

Delete a math problem and its associated files (image and audio).

**curl Example:**
```bash
curl -X DELETE http://localhost:5000/problems/550e8400-e29b-41d4-a716-446655440000
```

**Response (200 OK):**
```json
{
  "message": "Problem deleted successfully"
}
```

**Response (404 Not Found):**
```json
{"error": "Problem not found"}
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

## Mobile Web App Client Integration

### Complete Workflow Example

```javascript
const API_BASE = 'http://localhost:5000';

// 1. First, search if problem already exists
async function searchProblem(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    body: formData
  });
  const result = await response.json();
  return result.uuid;  // Returns UUID or null
}

// 2. If not found, create new problem
async function createProblem(imageFile, solutionLatex, audioFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('solution_latex', solutionLatex);
  formData.append('audio', audioFile);

  const response = await fetch(`${API_BASE}/problems`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error);
  }

  const result = await response.json();
  return result.uuid;
}

// 3. Get solution and play audio
async function getSolution(uuid) {
  const response = await fetch(`${API_BASE}/problems/${uuid}`);
  const problem = await response.json();

  // Display LaTeX solution (use a library like MathJax or KaTeX)
  displayLatex(problem.solution_latex);

  // Play audio explanation
  const audio = new Audio(`${API_BASE}${problem.audio_url}`);
  audio.play();
}

// Complete workflow
async function handleMathProblem(imageFile, solutionLatex, audioFile) {
  // Check if problem exists
  let uuid = await searchProblem(imageFile);

  if (uuid) {
    console.log('Problem found:', uuid);
  } else {
    // Create new problem
    uuid = await createProblem(imageFile, solutionLatex, audioFile);
    console.log('Problem created:', uuid);
  }

  // Get and display solution
  await getSolution(uuid);
}
```

### HTML Form Example

```html
<form id="mathProblemForm">
  <div>
    <label>Math Problem Image:</label>
    <input type="file" name="image" accept="image/*" required>
  </div>
  <div>
    <label>Solution (LaTeX):</label>
    <textarea name="solution_latex" placeholder="x = \frac{-b}{2a}" required></textarea>
  </div>
  <div>
    <label>Audio Explanation:</label>
    <input type="file" name="audio" accept="audio/*" required>
  </div>
  <button type="submit">Submit</button>
</form>

<script>
document.getElementById('mathProblemForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);

  const response = await fetch('http://localhost:5000/problems', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  alert('Created problem: ' + result.uuid);
});
</script>
```

## License

MIT
