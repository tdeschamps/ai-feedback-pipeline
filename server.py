"""
FastAPI server for the AI feedback categorization pipeline.
"""

import logging
import time


try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    logging.warning("FastAPI not available - server functionality disabled")
    FASTAPI_AVAILABLE = False

    # Create dummy classes for testing
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FastAPI:
        def __init__(self, **kwargs):
            pass

    def Field(**kwargs):
        return None


try:
    from pipeline import FeedbackPipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    logging.warning("Pipeline not available - core functionality disabled")
    PIPELINE_AVAILABLE = False

    # Create dummy class for testing
    class FeedbackPipeline:
        def __init__(self):
            pass

        async def sync_notion_problems(self):
            return True

        async def process_transcript(self, content, transcript_id):
            return {"status": "mock", "feedbacks_extracted": 0}

        async def batch_process_transcripts(self, transcripts):
            return {"total_transcripts": 0, "results": []}


logger = logging.getLogger(__name__)


# Pydantic Models
if FASTAPI_AVAILABLE:
    # Use real Pydantic models when FastAPI is available
    class WebhookPayload(BaseModel):
        """Webhook payload for transcript processing."""

        event_type: str
        transcript_id: str
        transcript_content: str
        metadata: dict = Field(default_factory=dict)

    class TranscriptData(BaseModel):
        """Individual transcript data."""

        id: str
        content: str

    class BatchTranscriptRequest(BaseModel):
        """Request model for batch transcript processing."""

        transcripts: list[TranscriptData]
        metadata: dict = Field(default_factory=dict)

    class ProcessingResponse(BaseModel):
        """Response model for individual transcript processing."""

        transcript_id: str
        feedbacks_extracted: int = 0
        matches_found: int = 0
        problems_updated: int = 0
        status: str = "unknown"
        error_message: str | None = None
        processing_time: float | None = None

    class BatchProcessingResponse(BaseModel):
        """Response model for batch processing."""

        total_transcripts: int = 0
        total_feedbacks: int = 0
        total_matches: int = 0
        total_updates: int = 0
        success_rate: float = 0.0
        processing_time: float = 0.0
        results: list[ProcessingResponse] = Field(default_factory=list)
        errors: dict | None = None

    class FeedbackRequest(BaseModel):
        """Request model for direct feedback processing."""

        content: str = Field(..., description="The feedback content/text")
        confidence: float = Field(
            ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
        )
        type: str = Field(..., description="Feedback type: feature_request or customer_pain")

        model_config = ConfigDict(
            json_schema_extra={
                "example": {
                    "content": "We need to export data to Excel files",
                    "confidence": 0.85,
                    "type": "feature_request",
                }
            }
        )

    class FeedbackResponse(BaseModel):
        """Response model for feedback processing."""

        feedback_id: str
        type: str
        content: str
        confidence: float
        match_found: bool = False
        problem_id: str | None = None
        problem_title: str | None = None
        match_confidence: float | None = None
        similarity_score: float | None = None
        reasoning: str | None = None
        status: str = "completed"
        error_message: str | None = None
        processing_time: float | None = None

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str = "unknown"
        version: str = "1.0.0"
        dependencies: dict = Field(default_factory=dict)

else:
    # Use simple fallback models when FastAPI is not available
    class WebhookPayload(BaseModel):
        """Webhook payload for transcript processing."""

        def __init__(
            self,
            event_type=None,
            transcript_id=None,
            transcript_content=None,
            metadata=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.event_type = event_type
            self.transcript_id = transcript_id
            self.transcript_content = transcript_content
            self.metadata = metadata

    class TranscriptData(BaseModel):
        """Individual transcript data."""

        def __init__(self, id=None, content=None, **kwargs):
            super().__init__(**kwargs)
            self.id = id
            self.content = content

    class BatchTranscriptRequest(BaseModel):
        """Request model for batch transcript processing."""

        def __init__(self, transcripts=None, metadata=None, **kwargs):
            super().__init__(**kwargs)
            self.transcripts = transcripts or []
            self.metadata = metadata

    class ProcessingResponse(BaseModel):
        """Response model for individual transcript processing."""

        def __init__(
            self,
            transcript_id=None,
            feedbacks_extracted=0,
            matches_found=0,
            problems_updated=0,
            status="unknown",
            error_message=None,
            processing_time=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.transcript_id = transcript_id
            self.feedbacks_extracted = feedbacks_extracted
            self.matches_found = matches_found
            self.problems_updated = problems_updated
            self.status = status
            self.error_message = error_message
            self.processing_time = processing_time

    class BatchProcessingResponse(BaseModel):
        """Response model for batch processing."""

        def __init__(
            self,
            total_transcripts=0,
            total_feedbacks=0,
            total_matches=0,
            total_updates=0,
            success_rate=0.0,
            processing_time=0.0,
            results=None,
            errors=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.total_transcripts = total_transcripts
            self.total_feedbacks = total_feedbacks
            self.total_matches = total_matches
            self.total_updates = total_updates
            self.success_rate = success_rate
            self.processing_time = processing_time
            self.results = results or []
            self.errors = errors

    class FeedbackRequest(BaseModel):
        """Request model for direct feedback processing."""

        def __init__(self, content=None, confidence=None, type=None, **kwargs):
            super().__init__(**kwargs)
            self.content = content
            self.confidence = confidence
            self.type = type

    class FeedbackResponse(BaseModel):
        """Response model for feedback processing."""

        def __init__(
            self,
            feedback_id=None,
            type=None,
            content=None,
            confidence=None,
            match_found=False,
            problem_id=None,
            problem_title=None,
            match_confidence=None,
            similarity_score=None,
            reasoning=None,
            status="completed",
            error_message=None,
            processing_time=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.feedback_id = feedback_id
            self.type = type
            self.content = content
            self.confidence = confidence
            self.match_found = match_found
            self.problem_id = problem_id
            self.problem_title = problem_title
            self.match_confidence = match_confidence
            self.similarity_score = similarity_score
            self.reasoning = reasoning
            self.status = status
            self.error_message = error_message
            self.processing_time = processing_time

    class HealthResponse(BaseModel):
        """Health check response."""

        def __init__(self, status="unknown", version="1.0.0", dependencies=None, **kwargs):
            super().__init__(**kwargs)
            self.status = status
            self.version = version
            self.dependencies = dependencies or {}


# Global pipeline instance
_pipeline: FeedbackPipeline | None = None


async def get_pipeline() -> FeedbackPipeline:
    """Dependency to get the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FeedbackPipeline()
        # Ensure Notion problems are synced
        await _pipeline.sync_notion_problems()
    return _pipeline


# FastAPI app
app = None

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AI Feedback Categorization Pipeline",
        description="API for processing customer feedback and matching to product problems",
        version="1.0.0",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            await get_pipeline()
            return HealthResponse(
                status="healthy",
                dependencies={
                    "pipeline": "operational",
                    "embedding_manager": "operational",
                    "notion_client": "operational",
                },
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unavailable")

    @app.post("/webhook/transcript", response_model=ProcessingResponse)
    async def process_transcript_webhook(
        payload: WebhookPayload, pipeline: FeedbackPipeline = Depends(get_pipeline)
    ):
        """Process a single transcript via webhook."""
        start_time = time.time()

        try:
            logger.info(f"Processing webhook for transcript {payload.transcript_id}")

            result = await pipeline.process_transcript(
                payload.transcript_content, payload.transcript_id
            )

            processing_time = time.time() - start_time

            return ProcessingResponse(
                transcript_id=payload.transcript_id,
                feedbacks_extracted=result.get("feedbacks_extracted", 0),
                matches_found=result.get("matches_found", 0),
                problems_updated=result.get("problems_updated", 0),
                status=result.get("status", "completed"),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing transcript {payload.transcript_id}: {e}")

            return ProcessingResponse(
                transcript_id=payload.transcript_id,
                feedbacks_extracted=0,
                matches_found=0,
                problems_updated=0,
                status="error",
                error_message=str(e),
                processing_time=processing_time,
            )

    @app.post("/process/batch", response_model=BatchProcessingResponse)
    async def process_batch_transcripts(
        request: BatchTranscriptRequest, pipeline: FeedbackPipeline = Depends(get_pipeline)
    ):
        """Process multiple transcripts in batch."""
        start_time = time.time()

        try:
            logger.info(f"Processing batch of {len(request.transcripts)} transcripts")

            # Convert to pipeline format
            transcripts_data = [{"id": t.id, "content": t.content} for t in request.transcripts]

            result = await pipeline.batch_process_transcripts(transcripts_data)

            processing_time = time.time() - start_time

            # Convert results to response format
            individual_results = []
            for transcript_result in result.get("results", []):
                individual_results.append(
                    ProcessingResponse(
                        transcript_id=transcript_result.get("transcript_id", "unknown"),
                        feedbacks_extracted=transcript_result.get("feedbacks_extracted", 0),
                        matches_found=transcript_result.get("matches_found", 0),
                        problems_updated=transcript_result.get("problems_updated", 0),
                        status=transcript_result.get("status", "completed"),
                        error_message=transcript_result.get("error"),
                    )
                )

            return BatchProcessingResponse(
                total_transcripts=result.get("total_transcripts", 0),
                total_feedbacks=result.get("total_feedbacks", 0),
                total_matches=result.get("total_matches", 0),
                total_updates=result.get("total_updates", 0),
                success_rate=result.get("success_rate", 0.0),
                processing_time=processing_time,
                results=individual_results,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing batch: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sync/notion")
    async def sync_notion_problems(pipeline: FeedbackPipeline = Depends(get_pipeline)):
        """Manually trigger Notion problems sync."""
        try:
            logger.info("Manual Notion sync requested")
            success = await pipeline.sync_notion_problems()

            if success:
                return {"status": "success", "message": "Notion problems synced successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to sync Notion problems")

        except Exception as e:
            logger.error(f"Error syncing Notion problems: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/process/feedback", response_model=FeedbackResponse)
    async def process_feedback(
        request: FeedbackRequest, pipeline: FeedbackPipeline = Depends(get_pipeline)
    ):
        """Process a single feedback directly."""
        start_time = time.time()

        try:
            # Validate feedback type
            if request.type not in ["feature_request", "customer_pain"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid feedback type. Must be 'feature_request' or 'customer_pain'",
                )

            # Validate confidence
            if not (0.0 <= request.confidence <= 1.0):
                raise HTTPException(
                    status_code=400, detail="Confidence must be between 0.0 and 1.0"
                )

            # Validate content
            if not request.content.strip():
                raise HTTPException(status_code=400, detail="Content cannot be empty")

            logger.info(f"Processing direct feedback of type {request.type}")

            # Create Feedback object
            import uuid
            from datetime import datetime

            from extract import Feedback

            feedback_id = str(uuid.uuid4())
            feedback = Feedback(
                type=request.type,
                summary=request.content[:100] + "..."
                if len(request.content) > 100
                else request.content,
                verbatim=request.content,
                confidence=request.confidence,
                transcript_id=f"api-{feedback_id}",
                timestamp=datetime.now(),
                context="API request",
            )

            # Process feedback through pipeline
            results = await pipeline.process_feedbacks([feedback])

            processing_time = time.time() - start_time

            # Extract result
            feedback_obj, match = results[0] if results else (feedback, None)

            return FeedbackResponse(
                feedback_id=feedback_id,
                type=feedback_obj.type,
                content=feedback_obj.verbatim,
                confidence=feedback_obj.confidence,
                match_found=match is not None,
                problem_id=match.problem_id if match else None,
                problem_title=match.problem_title if match else None,
                match_confidence=match.confidence if match else None,
                similarity_score=match.similarity_score if match else None,
                reasoning=match.reasoning if match else None,
                status="completed",
                processing_time=processing_time,
            )

        except HTTPException:
            # Re-raise HTTPExceptions as-is
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing feedback: {e}")

            return FeedbackResponse(
                feedback_id=str(uuid.uuid4()) if "feedback_id" not in locals() else feedback_id,
                type=request.type,
                content=request.content,
                confidence=request.confidence,
                match_found=False,
                status="error",
                error_message=str(e),
                processing_time=processing_time,
            )

    @app.get("/metrics")
    async def get_metrics(pipeline: FeedbackPipeline = Depends(get_pipeline)):
        """Get pipeline metrics."""
        try:
            metrics = pipeline.metrics.get_statistics()
            return {"metrics": metrics}
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available - cannot run server")

    uvicorn.run("server:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    run_server()
