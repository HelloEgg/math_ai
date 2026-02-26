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
    latex_string = db.Column(db.Text, nullable=True, index=True)  # OCR extracted question text
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
            'latex_string': self.latex_string,
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
    latex_string = db.Column(db.Text, nullable=True, index=True)  # OCR extracted question text
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
            'latex_string': self.latex_string,
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
    latex_string = db.Column(db.Text, nullable=True, index=True)  # OCR extracted question text
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
            'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class MathProblemOriginal(db.Model):
    """Model for storing original math problems (image URL + solution + answer, no audio)."""

    __tablename__ = 'math_problems_original'
    __table_args__ = (
        db.UniqueConstraint('image_url', 'feature', name='uq_original_url_feature'),
    )

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, index=True)
    image_url = db.Column(db.String(2048), nullable=False)  # Original image URL from client
    image_path = db.Column(db.String(512), nullable=True)   # Local cached copy
    solution_latex = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)               # Answer string
    feature = db.Column(db.Text, nullable=True)               # Feature plain text
    latex_string = db.Column(db.Text, nullable=True, index=True)  # OCR extracted question text
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<MathProblemOriginal {self.id}>'

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_url': self.image_url,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'answer': self.answer,
            'feature': self.feature,
            'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class EnglishProblem(db.Model):
    """Model for storing English problems with solutions and audio."""

    __tablename__ = 'english_problems'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<EnglishProblem {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_url': self.image_url,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer,
            'feature': self.feature,
            'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ScienceProblem(db.Model):
    """Model for storing science problems with solutions and audio."""

    __tablename__ = 'science_problems'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<ScienceProblem {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_url': self.image_url,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer,
            'feature': self.feature,
            'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class SocialScienceProblem(db.Model):
    """Model for storing social science problems with solutions and audio."""

    __tablename__ = 'social_science_problems'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<SocialScienceProblem {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_url': self.image_url,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer,
            'feature': self.feature,
            'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class KoreanProblem(db.Model):
    """Model for storing Korean problems with solutions and audio."""

    __tablename__ = 'korean_problems'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<KoreanProblem {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'image_url': self.image_url,
            'image_path': self.image_path,
            'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer,
            'feature': self.feature,
            'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# =============================================================================
# Summary Models for Other Subjects
# =============================================================================

class EnglishProblemSummary(db.Model):
    """Model for storing English problem summaries with audio."""
    __tablename__ = 'english_problems_summary'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<EnglishProblemSummary {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ScienceProblemSummary(db.Model):
    """Model for storing science problem summaries with audio."""
    __tablename__ = 'science_problems_summary'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<ScienceProblemSummary {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class SocialScienceProblemSummary(db.Model):
    """Model for storing social science problem summaries with audio."""
    __tablename__ = 'social_science_problems_summary'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<SocialScienceProblemSummary {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class KoreanProblemSummary(db.Model):
    """Model for storing Korean problem summaries with audio."""
    __tablename__ = 'korean_problems_summary'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<KoreanProblemSummary {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# =============================================================================
# Deep Models for Other Subjects
# =============================================================================

class EnglishProblemDeep(db.Model):
    """Model for storing English problem deep solutions with audio."""
    __tablename__ = 'english_problems_deep'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<EnglishProblemDeep {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ScienceProblemDeep(db.Model):
    """Model for storing science problem deep solutions with audio."""
    __tablename__ = 'science_problems_deep'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<ScienceProblemDeep {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class SocialScienceProblemDeep(db.Model):
    """Model for storing social science problem deep solutions with audio."""
    __tablename__ = 'social_science_problems_deep'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<SocialScienceProblemDeep {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class KoreanProblemDeep(db.Model):
    """Model for storing Korean problem deep solutions with audio."""
    __tablename__ = 'korean_problems_deep'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    image_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    image_url = db.Column(db.String(2048), nullable=False)
    image_path = db.Column(db.String(512), nullable=True)
    solution_latex = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.String(512), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    feature = db.Column(db.Text, nullable=True)
    latex_string = db.Column(db.Text, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<KoreanProblemDeep {self.id}>'

    def to_dict(self):
        return {
            'id': self.id, 'image_hash': self.image_hash, 'image_url': self.image_url,
            'image_path': self.image_path, 'solution_latex': self.solution_latex,
            'audio_path': self.audio_path,
            'answer': self.answer, 'feature': self.feature, 'latex_string': self.latex_string,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
