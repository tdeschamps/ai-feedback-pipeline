"""
RAG (Retrieval-Augmented Generation) module for matching feedback to problems.
"""
import logging
from dataclasses import dataclass
from typing import Any

from config import settings
from embed import EmbeddingManager
from extract import Feedback
from llm_client import get_llm_client


logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Result of matching feedback to a problem."""
    problem_id: str
    problem_title: str
    confidence: float
    similarity_score: float
    reasoning: str
    metadata: dict[str, Any]

class RAGMatcher:
    """RAG-based feedback to problem matching."""

    def __init__(self) -> None:
        self.llm_client = get_llm_client()
        self.embedding_manager = EmbeddingManager()
        self.rerank_prompt = self._build_rerank_prompt()

    def _build_rerank_prompt(self) -> str:
        """Build prompt for LLM-based reranking."""
        return """You are an expert at matching customer feedback to existing product problems.

Your task is to analyze a piece of customer feedback and determine if it matches any of the candidate problems from our database.

Consider these factors:
1. SEMANTIC SIMILARITY: Does the feedback describe the same underlying issue?
2. PROBLEM SCOPE: Is the feedback addressing the same area/feature?
3. SEVERITY ALIGNMENT: Does the feedback intensity match the problem priority?
4. SOLUTION OVERLAP: Would solving the problem address the feedback?

For each candidate problem, provide:
- match_confidence: 0.0-1.0 (1.0 = perfect match, 0.0 = no match)
- reasoning: Brief explanation of why it matches or doesn't match

Return your analysis as a JSON object with this structure:
{
  "best_match": {
    "problem_id": "string",
    "confidence": 0.8,
    "reasoning": "Detailed explanation"
  },
  "all_matches": [
    {
      "problem_id": "string",
      "confidence": 0.8,
      "reasoning": "Brief explanation"
    }
  ]
}

If no problem has confidence >= 0.6, set best_match to null."""

    async def find_best_match(self, feedback: Feedback) -> MatchResult | None:
        """Find the best matching problem for a feedback."""
        try:
            # Step 1: Semantic search for candidate problems
            feedback_text = f"{feedback.summary}\n\n{feedback.verbatim}"
            candidates = await self.embedding_manager.search_similar_problems(
                feedback_text,
                limit=settings.max_matches
            )

            if not candidates:
                logger.info(f"No candidate problems found for feedback: {feedback.summary}")
                return None

            # Step 2: LLM-based reranking
            if settings.rerank_enabled:
                best_match = await self._llm_rerank(feedback, candidates)
            else:
                # Use the top semantic match
                best_match = await self._use_top_match(feedback, candidates)

            # Step 3: Apply confidence threshold
            if best_match and best_match.confidence >= settings.confidence_threshold:
                logger.info(f"Found match for feedback with confidence {best_match.confidence:.3f}")
                return best_match
            else:
                confidence_msg = f"{best_match.confidence:.3f}" if best_match else "0"
                logger.info(f"No high-confidence match found (best: {confidence_msg})")
                return None

        except Exception as e:
            logger.error(f"Error finding match for feedback: {e}")
            return None

    async def _llm_rerank(self, feedback: Feedback, candidates: list[dict[str, Any]]) -> MatchResult | None:
        """Use LLM to rerank and select best match."""
        try:
            # Prepare candidate information for LLM
            candidates_text = ""
            for i, candidate in enumerate(candidates):
                metadata = candidate.get("metadata", {})
                similarity = candidate.get("certainty", candidate.get("similarity", 0))

                candidates_text += f"\n=== CANDIDATE {i+1} ===\n"
                candidates_text += f"ID: {metadata.get('notion_id', 'unknown')}\n"
                candidates_text += f"Title: {metadata.get('title', 'No title')}\n"
                candidates_text += f"Status: {metadata.get('status', 'Unknown')}\n"
                candidates_text += f"Priority: {metadata.get('priority', 'Unknown')}\n"
                candidates_text += f"Semantic Similarity: {similarity:.3f}\n"
                candidates_text += f"Content: {candidate.get('content', '')[:300]}...\n"

            # Prepare the prompt
            messages = [
                {"role": "system", "content": self.rerank_prompt},
                {"role": "user", "content": f"""
FEEDBACK TO MATCH:
Type: {feedback.type}
Summary: {feedback.summary}
Verbatim: "{feedback.verbatim}"
Confidence: {feedback.confidence}

CANDIDATE PROBLEMS:
{candidates_text}

Please analyze and return your matching assessment as JSON."""}
            ]

            response = await self.llm_client.generate(messages)
            result = self._parse_rerank_response(response.content, candidates)

            return result

        except Exception as e:
            logger.error(f"Error in LLM reranking: {e}")
            return None

    def _parse_rerank_response(self, response: str, candidates: list[dict[str, Any]]) -> MatchResult | None:
        """Parse LLM reranking response."""
        try:
            import json

            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON found in rerank response")
                return None

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            best_match = data.get("best_match")
            if not best_match or best_match.get("confidence", 0) < 0.6:
                return None

            # Find the corresponding candidate
            problem_id = best_match["problem_id"]
            matching_candidate = None

            for candidate in candidates:
                if candidate.get("metadata", {}).get("notion_id") == problem_id:
                    matching_candidate = candidate
                    break

            if not matching_candidate:
                logger.warning(f"Could not find candidate for problem_id: {problem_id}")
                return None

            metadata = matching_candidate.get("metadata", {})
            similarity = matching_candidate.get("certainty", matching_candidate.get("similarity", 0))

            return MatchResult(
                problem_id=problem_id,
                problem_title=metadata.get("title", "Unknown"),
                confidence=best_match["confidence"],
                similarity_score=similarity,
                reasoning=best_match.get("reasoning", ""),
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error parsing rerank response: {e}")
            return None

    async def _use_top_match(self, feedback: Feedback, candidates: list[dict[str, Any]]) -> MatchResult | None:
        """Use top semantic match without LLM reranking."""
        if not candidates:
            return None

        top_candidate = candidates[0]
        metadata = top_candidate.get("metadata", {})
        similarity = top_candidate.get("certainty", top_candidate.get("similarity", 0))

        # Convert similarity to confidence (simple mapping)
        confidence = min(similarity * 1.2, 1.0)  # Boost similarity slightly

        return MatchResult(
            problem_id=metadata.get("notion_id", "unknown"),
            problem_title=metadata.get("title", "Unknown"),
            confidence=confidence,
            similarity_score=similarity,
            reasoning="Top semantic match (no LLM reranking)",
            metadata=metadata
        )

    async def batch_match(self, feedbacks: list[Feedback]) -> list[tuple[Feedback, MatchResult | None]]:
        """Match multiple feedbacks to problems."""
        results = []

        for feedback in feedbacks:
            match = await self.find_best_match(feedback)
            results.append((feedback, match))

        matched_count = sum(1 for _, match in results if match)
        logger.info(f"Matched {matched_count}/{len(feedbacks)} feedbacks to problems")

        return results

class MatchingMetrics:
    """Track and analyze matching performance."""

    def __init__(self) -> None:
        self.matches: list[dict[str, Any]] = []
        self.unmatched: list[dict[str, Any]] = []

    def add_result(self, feedback: Feedback, match: MatchResult | None) -> None:
        """Add a matching result."""
        if match:
            self.matches.append({
                "feedback": feedback,
                "match": match,
                "timestamp": feedback.timestamp.isoformat()
            })
        else:
            self.unmatched.append({
                "feedback": feedback,
                "timestamp": feedback.timestamp.isoformat()
            })

    def get_statistics(self) -> dict[str, Any]:
        """Get matching statistics."""
        total = len(self.matches) + len(self.unmatched)

        if total == 0:
            return {"total": 0, "match_rate": 0, "avg_confidence": 0}

        match_rate = len(self.matches) / total

        confidences = [match["match"].confidence for match in self.matches]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Confidence distribution
        confidence_buckets = {"high": 0, "medium": 0, "low": 0}
        for conf in confidences:
            if conf >= 0.8:
                confidence_buckets["high"] += 1
            elif conf >= 0.6:
                confidence_buckets["medium"] += 1
            else:
                confidence_buckets["low"] += 1

        return {
            "total_feedbacks": total,
            "matched": len(self.matches),
            "unmatched": len(self.unmatched),
            "match_rate": match_rate,
            "avg_confidence": avg_confidence,
            "confidence_distribution": confidence_buckets
        }

    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file."""
        import json

        data = {
            "statistics": self.get_statistics(),
            "matches": [
                {
                    "feedback_summary": match["feedback"].summary,
                    "problem_title": match["match"].problem_title,
                    "confidence": match["match"].confidence,
                    "timestamp": match["timestamp"]
                }
                for match in self.matches
            ],
            "unmatched": [
                {
                    "feedback_summary": item["feedback"].summary,
                    "feedback_type": item["feedback"].type,
                    "timestamp": item["timestamp"]
                }
                for item in self.unmatched
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved matching metrics to {filepath}")
