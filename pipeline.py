"""
High-level pipeline orchestration for the feedback categorization system.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings, setup_logging
from embed import EmbeddingManager
from extract import Feedback, FeedbackExtractor, save_feedback_logs
from notion import NotionClient
from rag import MatchingMetrics, RAGMatcher


logger = logging.getLogger(__name__)


class FeedbackPipeline:
    """Main pipeline for processing feedback and matching to problems."""

    def __init__(self) -> None:
        self.extractor = FeedbackExtractor()
        self.notion_client = NotionClient()
        self.embedding_manager = EmbeddingManager()
        self.matcher = RAGMatcher()
        self.metrics = MatchingMetrics()

    async def process_transcript(self, transcript: str, transcript_id: str) -> dict[str, Any]:
        """Process a single transcript through the entire pipeline."""
        logger.info(f"Processing transcript: {transcript_id}")

        try:
            # Step 1: Extract feedback
            feedbacks = await self.extractor.extract_feedback(transcript, transcript_id)

            if not feedbacks:
                logger.info(f"No feedback extracted from transcript {transcript_id}")
                return {
                    "transcript_id": transcript_id,
                    "feedbacks_extracted": 0,
                    "matches_found": 0,
                    "problems_updated": 0,
                    "status": "no_feedback",
                }

            # Step 2: Process each feedback
            results = await self.process_feedbacks(feedbacks)

            # Step 3: Save logs
            await self._save_processing_logs(transcript_id, feedbacks, results)

            # Compile results
            matches_found = sum(1 for _, match in results if match)
            problems_updated = sum(
                1
                for _, match in results
                if match and match.confidence >= settings.confidence_threshold
            )

            return {
                "transcript_id": transcript_id,
                "feedbacks_extracted": len(feedbacks),
                "matches_found": matches_found,
                "problems_updated": problems_updated,
                "status": "completed",
                "results": results,
            }

        except Exception as e:
            logger.error(f"Error processing transcript {transcript_id}: {e}")
            return {"transcript_id": transcript_id, "error": str(e), "status": "error"}

    async def process_feedbacks(self, feedbacks: list[Feedback]) -> list[tuple[Feedback, Any]]:
        """Process extracted feedbacks through matching and updating."""
        results = []

        for feedback in feedbacks:
            try:
                # Find best match
                match = await self.matcher.find_best_match(feedback)

                # Track metrics
                self.metrics.add_result(feedback, match)

                if match and match.confidence >= settings.confidence_threshold:
                    # Update Notion problem
                    success = self.notion_client.update_problem_with_feedback(
                        match.problem_id, feedback, match.confidence
                    )

                    if success:
                        logger.info(f"Updated problem {match.problem_title} with feedback")
                    else:
                        logger.warning(f"Failed to update problem {match.problem_id}")

                elif match:
                    logger.info(
                        f"Low confidence match ({match.confidence:.3f}) - logging for review"
                    )

                else:
                    logger.info(f"No match found for feedback: {feedback.summary[:50]}...")

                results.append((feedback, match))

            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
                results.append((feedback, None))

        return results

    async def batch_process_transcripts(self, transcripts: list[dict[str, str]]) -> dict[str, Any]:
        """Process multiple transcripts in batch."""
        logger.info(f"Starting batch processing of {len(transcripts)} transcripts")

        # Ensure problems are synced
        await self.sync_notion_problems()

        results: list[dict[str, Any]] = []
        total_feedbacks = 0
        total_matches = 0
        total_updates = 0

        for transcript_data in transcripts:
            transcript_id = transcript_data.get("id", f"unknown_{len(results)}")
            transcript_content = transcript_data.get("content", "")

            if not transcript_content:
                logger.warning(f"Empty transcript content for {transcript_id}")
                continue

            result = await self.process_transcript(transcript_content, transcript_id)
            results.append(result)

            total_feedbacks += result.get("feedbacks_extracted", 0)
            total_matches += result.get("matches_found", 0)
            total_updates += result.get("problems_updated", 0)

        # Save overall metrics
        metrics_file = f"data/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.metrics.save_metrics(metrics_file)

        return {
            "total_transcripts": len(transcripts),
            "total_feedbacks": total_feedbacks,
            "total_matches": total_matches,
            "total_updates": total_updates,
            "success_rate": total_matches / total_feedbacks if total_feedbacks > 0 else 0,
            "results": results,
            "metrics_file": metrics_file,
        }

    async def sync_notion_problems(self) -> bool:
        """Sync and embed all Notion problems."""
        try:
            logger.info("Syncing Notion problems...")

            # Get all problems from Notion
            problems = self.notion_client.get_all_problems()

            if not problems:
                logger.warning("No problems retrieved from Notion")
                return False

            # Refresh embeddings
            success = await self.embedding_manager.refresh_problem_embeddings(problems)

            if success:
                logger.info(f"Successfully synced {len(problems)} problems")
            else:
                logger.error("Failed to sync problem embeddings")

            return success

        except Exception as e:
            logger.error(f"Error syncing Notion problems: {e}")
            return False

    async def _save_processing_logs(
        self, transcript_id: str, feedbacks: list[Feedback], results: list[tuple[Feedback, Any]]
    ) -> None:
        """Save processing logs for audit trail."""
        try:
            # Save feedback logs
            feedback_log_file = f"data/feedback_logs_{datetime.now().strftime('%Y%m%d')}.json"
            save_feedback_logs(feedbacks, feedback_log_file)

            # Save detailed results
            import json

            detailed_results = []
            for feedback, match in results:
                result_data = {
                    "feedback": {
                        "type": feedback.type,
                        "summary": feedback.summary,
                        "verbatim": feedback.verbatim,
                        "confidence": feedback.confidence,
                        "transcript_id": feedback.transcript_id,
                        "timestamp": feedback.timestamp.isoformat(),
                    },
                    "match": (
                        {
                            "problem_id": match.problem_id if match else None,
                            "problem_title": match.problem_title if match else None,
                            "confidence": match.confidence if match else None,
                            "similarity_score": match.similarity_score if match else None,
                            "reasoning": match.reasoning if match else None,
                        }
                        if match
                        else None
                    ),
                }
                detailed_results.append(result_data)

            results_file = f"data/processing_results_{transcript_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved processing logs to {results_file}")

        except Exception as e:
            logger.error(f"Error saving processing logs: {e}")


async def run_pipeline_cli() -> None:
    """CLI runner for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Feedback Categorization Pipeline")
    parser.add_argument("--transcript-file", help="Path to transcript file")
    parser.add_argument("--transcript-dir", help="Directory containing transcript files")
    parser.add_argument("--sync-only", action="store_true", help="Only sync Notion problems")
    parser.add_argument("--transcript-id", help="Transcript ID (if processing single file)")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Initialize pipeline
    pipeline = FeedbackPipeline()

    try:
        if args.sync_only:
            logger.info("Syncing Notion problems only...")
            success = await pipeline.sync_notion_problems()
            print(f"Sync {'successful' if success else 'failed'}")
            return

        if args.transcript_file:
            # Process single file
            with open(args.transcript_file, encoding="utf-8") as f:
                transcript_content = f.read()

            transcript_id = args.transcript_id or Path(args.transcript_file).stem
            result = await pipeline.process_transcript(transcript_content, transcript_id)

            print(f"Processed transcript {transcript_id}:")
            print(f"  Feedbacks extracted: {result['feedbacks_extracted']}")
            print(f"  Matches found: {result['matches_found']}")
            print(f"  Problems updated: {result['problems_updated']}")

        elif args.transcript_dir:
            # Process directory
            transcript_dir = Path(args.transcript_dir)
            transcripts = []

            for file_path in transcript_dir.glob("*.txt"):
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                transcripts.append({"id": file_path.stem, "content": content})

            if not transcripts:
                print(f"No .txt files found in {transcript_dir}")
                return

            result = await pipeline.batch_process_transcripts(transcripts)

            print("Batch processing completed:")
            print(f"  Transcripts processed: {result['total_transcripts']}")
            print(f"  Total feedbacks: {result['total_feedbacks']}")
            print(f"  Total matches: {result['total_matches']}")
            print(f"  Total updates: {result['total_updates']}")
            print(f"  Success rate: {result['success_rate']:.2%}")
            print(f"  Metrics saved to: {result['metrics_file']}")

        else:
            print("Please specify --transcript-file or --transcript-dir")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_pipeline_cli())
