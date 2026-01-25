import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def generate_uuid():
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class MathProblem(db.Model):
    """Model for storing math problems with solutions and explanations."""

    __tablename__ = 'math_problems'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_path = db.Column(db.String(512), nullable=False)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<MathProblem {self.id}>'

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class MathProblemSummary(db.Model):
    """Model for storing math problem summaries with solutions and explanations."""

    __tablename__ = 'math_problems_summary'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_path = db.Column(db.String(512), nullable=False)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<MathProblemSummary {self.id}>'

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class MathProblemDeep(db.Model):
    """Model for storing deep math problem explanations with solutions."""

    __tablename__ = 'math_problems_deep'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_path = db.Column(db.String(512), nullable=False)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<MathProblemDeep {self.id}>'

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
