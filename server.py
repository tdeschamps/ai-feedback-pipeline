"""
FastAPI server for the feedback pipeline with webhook support.
"""
import logging
from datetime import datetime
from typing import Any


try:
    import uvicorn
    from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    logging.warning("FastAPI dependencies not available")

from config import settings, setup_logging
from pipeline import FeedbackPipeline


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Pydantic models
class TranscriptRequest(BaseModel):
    """Request model for single transcript processing."""
    transcript_id: str = Field(..., description="Unique identifier for the transcript")
    content: str = Field(..., description="Transcript content to process")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")

class BatchTranscriptRequest(BaseModel):
    """Request model for batch transcript processing."""
    transcripts: list[dict[str, str]] = Field(..., description="List of transcripts with id and content")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")

class ProcessingResponse(BaseModel):
    """Response model for processing results."""
    transcript_id: str
    feedbacks_extracted: int
    matches_found: int
    problems_updated: int
    status: str
    processing_time: float | None = None
    error: str | None = None

class BatchProcessingResponse(BaseModel):
    """Response model for batch processing results."""
    total_transcripts: int
    total_feedbacks: int
    total_matches: int
    total_updates: int
    success_rate: float
    processing_time: float
    results: list[ProcessingResponse]
    metrics_file: str | None = None

class WebhookPayload(BaseModel):
    """Webhook payload from external systems."""
    event_type: str = Field(..., description="Type of event (e.g., 'transcript_ready')")
    transcript_id: str = Field(..., description="Transcript identifier")
    transcript_url: str | None = Field(default=None, description="URL to download transcript")
    transcript_content: str | None = Field(default=None, description="Direct transcript content")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional event metadata")

# Initialize FastAPI app
app = FastAPI(
    title="AI Feedback Categorization Pipeline",
    description="RAG-based pipeline for categorizing customer feedback and matching to problems",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: FeedbackPipeline | None = None

async def get_pipeline() -> FeedbackPipeline:
    """Dependency to get pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = FeedbackPipeline()
    return pipeline

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize pipeline on startup."""
    global pipeline
    try:
        logger.info("Initializing pipeline...")
        pipeline = FeedbackPipeline()

        # Optionally sync problems on startup
        logger.info("Syncing Notion problems...")
        await pipeline.sync_notion_problems()

        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "llm_provider": settings.llm_provider,
        "vector_store": settings.vector_store
    }

@app.post("/process/transcript", response_model=ProcessingResponse)
async def process_transcript(
    request: TranscriptRequest,
    background_tasks: BackgroundTasks,
    pipeline: FeedbackPipeline = Depends(get_pipeline)
) -> ProcessingResponse:
    """Process a single transcript."""
    start_time = datetime.now()

    try:
        logger.info(f"API: Processing transcript {request.transcript_id}")

        result = await pipeline.process_transcript(
            request.content,
            request.transcript_id
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return ProcessingResponse(
            transcript_id=result["transcript_id"],
            feedbacks_extracted=result["feedbacks_extracted"],
            matches_found=result["matches_found"],
            problems_updated=result["problems_updated"],
            status=result["status"],
            processing_time=processing_time,
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"API: Error processing transcript {request.transcript_id}: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()

        return ProcessingResponse(
            transcript_id=request.transcript_id,
            feedbacks_extracted=0,
            matches_found=0,
            problems_updated=0,
            status="error",
            processing_time=processing_time,
            error=str(e)
        )

@app.post("/process/batch", response_model=BatchProcessingResponse)
async def process_batch(
    request: BatchTranscriptRequest,
    background_tasks: BackgroundTasks,
    pipeline: FeedbackPipeline = Depends(get_pipeline)
) -> BatchProcessingResponse:
    """Process multiple transcripts in batch."""
    start_time = datetime.now()

    try:
        logger.info(f"API: Processing batch of {len(request.transcripts)} transcripts")

        result = await pipeline.batch_process_transcripts(request.transcripts)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Convert individual results
        individual_results = []
        for res in result.get("results", []):
            individual_results.append(ProcessingResponse(
                transcript_id=res["transcript_id"],
                feedbacks_extracted=res["feedbacks_extracted"],
                matches_found=res["matches_found"],
                problems_updated=res["problems_updated"],
                status=res["status"],
                error=res.get("error")
            ))

        return BatchProcessingResponse(
            total_transcripts=result["total_transcripts"],
            total_feedbacks=result["total_feedbacks"],
            total_matches=result["total_matches"],
            total_updates=result["total_updates"],
            success_rate=result["success_rate"],
            processing_time=processing_time,
            results=individual_results,
            metrics_file=result["metrics_file"]
        )

    except Exception as e:
        logger.error(f"API: Error processing batch: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()

        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        ) from e

@app.post("/webhook/circleback")
async def circleback_webhook(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    pipeline: FeedbackPipeline = Depends(get_pipeline)
) -> dict[str, str]:
    """Webhook endpoint for Circleback integration."""
    try:
        logger.info(f"Webhook: Received {payload.event_type} for transcript {payload.transcript_id}")

        if payload.event_type != "transcript_ready":
            return {"status": "ignored", "reason": f"Unknown event type: {payload.event_type}"}

        # Get transcript content
        transcript_content = payload.transcript_content

        if not transcript_content and payload.transcript_url:
            # Download transcript from URL
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(payload.transcript_url)
                response.raise_for_status()
                transcript_content = response.text

        if not transcript_content:
            raise HTTPException(
                status_code=400,
                detail="No transcript content provided"
            )

        # Process in background
        background_tasks.add_task(
            process_webhook_transcript,
            pipeline,
            transcript_content,
            payload.transcript_id,
            payload.metadata or {}
        )

        return {
            "status": "accepted",
            "transcript_id": payload.transcript_id,
            "message": "Transcript queued for processing"
        }

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

async def process_webhook_transcript(
    pipeline: FeedbackPipeline,
    content: str,
    transcript_id: str,
    metadata: dict[str, Any]
) -> None:
    """Background task to process webhook transcript."""
    try:
        logger.info(f"Background: Processing webhook transcript {transcript_id}")

        result = await pipeline.process_transcript(content, transcript_id)

        logger.info(f"Background: Completed processing {transcript_id} - "
                   f"{result['feedbacks_extracted']} feedbacks, "
                   f"{result['matches_found']} matches")

    except Exception as e:
        logger.error(f"Background: Error processing {transcript_id}: {e}")

@app.post("/sync/problems")
async def sync_problems(pipeline: FeedbackPipeline = Depends(get_pipeline)) -> dict[str, str]:
    """Manually trigger Notion problems sync."""
    try:
        logger.info("API: Syncing Notion problems")

        success = await pipeline.sync_notion_problems()

        if success:
            return {"status": "success", "message": "Problems synced successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to sync problems")

    except Exception as e:
        logger.error(f"API: Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/metrics")
async def get_metrics(pipeline: FeedbackPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    """Get current pipeline metrics."""
    try:
        metrics = pipeline.metrics.get_statistics()
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"API: Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Get current pipeline configuration."""
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "vector_store": settings.vector_store,
        "confidence_threshold": settings.confidence_threshold,
        "rerank_enabled": settings.rerank_enabled,
        "max_matches": settings.max_matches
    }

def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:  # noqa: S104
    """Run the FastAPI server."""
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
