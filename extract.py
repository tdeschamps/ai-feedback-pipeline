"""
Feedback extraction and classification from Circleback transcripts.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_client import get_llm_client


logger = logging.getLogger(__name__)


@dataclass
class Feedback:
    """Structured feedback extracted from transcript."""

    type: str  # "feature_request" or "customer_pain"
    summary: str
    verbatim: str
    confidence: float
    transcript_id: str
    timestamp: datetime
    context: str | None = None

    def __post_init__(self) -> None:
        """Validate feedback data after initialization."""
        if self.type not in ["feature_request", "customer_pain"]:
            raise ValueError(f"Invalid feedback type: {self.type}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {self.confidence}")

        if not self.summary.strip():
            raise ValueError("Summary cannot be empty")

        if not self.verbatim.strip():
            raise ValueError("Verbatim cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert feedback to dictionary format."""
        return {
            "type": self.type,
            "summary": self.summary,
            "verbatim": self.verbatim,
            "confidence": self.confidence,
            "transcript_id": self.transcript_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


class FeedbackExtractor:
    """Extract and classify feedback from transcripts using LLM."""

    def __init__(self) -> None:
        """Initialize the feedback extractor."""
        self.llm_client = get_llm_client()
        self.extraction_prompt = self._build_extraction_prompt()
        self._min_confidence = 0.3
        self._min_summary_length = 10
        self._min_verbatim_length = 5

    def _build_extraction_prompt(self) -> str:
        """Build the system prompt for feedback extraction."""
        return """Vous êtes un expert en analyse de conversations clients et en extraction de commentaires exploitables.

    Votre tâche est d'identifier et d'extraire deux types de commentaires à partir de transcriptions client :
    1. DEMANDES DE FONCTIONNALITÉS : Demandes explicites ou implicites de nouvelles fonctionnalités, améliorations ou capacités
    2. PROBLÈMES CLIENTS : Problèmes, frustrations, plaintes ou difficultés mentionnés par les clients

    Pour chaque commentaire trouvé, extrayez :
    - type: "feature_request" ou "customer_pain"
    - summary: Une description concise en 1-2 phrases du commentaire
    - verbatim: La citation exacte de la transcription (conservez la formulation originale)
    - confidence: Votre niveau de confiance (0.0-1.0) que ceci est un commentaire valide

    ### **Exemple de réponse attendue (format JSON)**
    [
    {
        "type": "feature_request",
        "summary": "Le client demande une intégration directe avec l'ERP pour éviter les retards de données.",
        "verbatim": "Il faudrait intégrer directement l'ERP pour éviter ces retards de données, car l'import en masse est trop lent.",
        "confidence": 0.95
    },
    {
        "type": "customer_pain",
        "summary": "Le client exprime des difficultés avec la scalabilité des outils Excel.",
        "verbatim": "Excel ne gère pas bien les processus complexes, surtout avec des équipes grandes.",
        "confidence": 0.9
    }
    ]
    """

    async def extract_feedback(self, transcript: str, transcript_id: str) -> list[Feedback]:
        """Extract structured feedback from a transcript."""
        try:
            messages = [
                {"role": "system", "content": self.extraction_prompt},
                {
                    "role": "user",
                    "content": f"Analysez cette transcription client et extrayez les commentaires :\n\n{transcript}",
                },
            ]

            response = await self.llm_client.generate(messages)
            logger.info(f"LLM response received for transcript {transcript_id}: {response}...")
            logger.info(
                f"LLM response received for transcript {transcript_id}: {response.content}..."
            )
            feedback_data = self._parse_feedback_response(response.content)

            # Convert to Feedback objects
            feedbacks = []
            for item in feedback_data:
                try:
                    feedback = Feedback(
                        type=item.get("type", "").lower(),
                        summary=item.get("summary", ""),
                        verbatim=item.get("verbatim", ""),
                        confidence=item.get("confidence", 0.0),
                        transcript_id=transcript_id,
                        timestamp=datetime.now(),
                        context=item.get("context"),
                    )

                    # Validate feedback
                    if self._is_valid_feedback(feedback):
                        feedbacks.append(feedback)
                    else:
                        logger.warning(f"Invalid feedback extracted: {feedback}")

                except ValueError as e:
                    logger.warning(f"Failed to create feedback object: {e}. Item: {item}")
                    continue

            logger.info(
                f"Extracted {len(feedbacks)} valid feedbacks from transcript {transcript_id}"
            )
            return feedbacks

        except Exception as e:
            logger.error(f"Error extracting feedback from transcript {transcript_id}: {e}")
            return []

    def _parse_feedback_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into structured feedback data."""
        try:
            # Try to extract JSON from response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON array found in response")
                return []

            json_str = response[start_idx:end_idx]
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return []

    def _is_valid_feedback(self, feedback: Feedback) -> bool:
        """Validate extracted feedback with improved checks."""
        try:
            # Type validation
            if feedback.type not in ["feature_request", "customer_pain"]:
                logger.debug(f"Invalid feedback type: {feedback.type}")
                return False

            # Content validation
            if not feedback.summary.strip() or not feedback.verbatim.strip():
                logger.debug("Empty summary or verbatim content")
                return False

            # Confidence validation
            if feedback.confidence < self._min_confidence:
                logger.debug(f"Low confidence: {feedback.confidence}")
                return False

            # Length validation
            if (
                len(feedback.summary.strip()) < self._min_summary_length
                or len(feedback.verbatim.strip()) < self._min_verbatim_length
            ):
                logger.debug("Content too short")
                return False

            # Quality checks
            if self._is_generic_feedback(feedback):
                logger.debug("Generic or low-quality feedback detected")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating feedback: {e}")
            return False

    def _is_generic_feedback(self, feedback: Feedback) -> bool:
        """Check if feedback is too generic or low quality."""
        generic_phrases = [
            "good job",
            "nice work",
            "thank you",
            "thanks",
            "great",
            "awesome",
            "perfect",
            "ok",
            "okay",
        ]

        summary_lower = feedback.summary.lower()

        # Check if feedback is mostly generic phrases
        for phrase in generic_phrases:
            if phrase in summary_lower and len(summary_lower.replace(phrase, "").strip()) < 5:
                return True

        return False

    async def batch_extract(self, transcripts: list[dict[str, str]]) -> list[Feedback]:
        """Extract feedback from multiple transcripts."""
        all_feedbacks = []

        for transcript_data in transcripts:
            transcript_id = transcript_data.get("id", "unknown")
            transcript_text = transcript_data.get("content", "")

            if not transcript_text:
                logger.warning(f"Empty transcript content for ID: {transcript_id}")
                continue

            feedbacks = await self.extract_feedback(transcript_text, transcript_id)
            all_feedbacks.extend(feedbacks)

        logger.info(
            f"Extracted {len(all_feedbacks)} total feedbacks from {len(transcripts)} transcripts"
        )
        return all_feedbacks


def save_feedback_logs(feedbacks: list[Feedback], filepath: str | Path) -> bool:
    """Save extracted feedbacks to JSON log file with error handling."""
    try:
        if not feedbacks:
            logger.warning("No feedbacks to save")
            return False

        # Ensure directory exists
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        feedback_data = []
        for feedback in feedbacks:
            try:
                feedback_dict = {
                    "type": feedback.type,
                    "summary": feedback.summary,
                    "verbatim": feedback.verbatim,
                    "confidence": feedback.confidence,
                    "transcript_id": feedback.transcript_id,
                    "timestamp": feedback.timestamp.isoformat(),
                    "context": feedback.context,
                }
                feedback_data.append(feedback_dict)
            except Exception as e:
                logger.error(f"Error serializing feedback: {e}")
                continue

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(feedback_data)} feedbacks to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error saving feedback logs to {filepath}: {e}")
        return False


def load_feedback_logs(filepath: str | Path) -> list[Feedback]:
    """Load feedbacks from JSON log file with error handling."""
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            logger.warning(f"Feedback log file not found: {filepath}")
            return []

        with open(file_path, encoding="utf-8") as f:
            feedback_data = json.load(f)

        if not isinstance(feedback_data, list):
            logger.error(f"Invalid feedback data format in {filepath}")
            return []

        feedbacks = []
        for item in feedback_data:
            try:
                feedback = Feedback(
                    type=item["type"],
                    summary=item["summary"],
                    verbatim=item["verbatim"],
                    confidence=item["confidence"],
                    transcript_id=item["transcript_id"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    context=item.get("context"),
                )
                feedbacks.append(feedback)
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Error parsing feedback item: {e}")
                continue

        logger.info(f"Loaded {len(feedbacks)} feedbacks from {filepath}")
        return feedbacks

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in feedback log file {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading feedback logs from {filepath}: {e}")
        return []
