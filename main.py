"""
Main CLI entry point for the AI Feedback Categorization Pipeline.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from config import settings, setup_logging
from pipeline import FeedbackPipeline


logger = logging.getLogger(__name__)


@click.group()
@click.option('--log-level', default='INFO', help='Logging level',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']))
@click.option('--config', help='Path to configuration file',
              type=click.Path(exists=True))
@click.pass_context
def cli(ctx: click.Context, log_level: str, config: str | None) -> None:
    """AI-Powered Feedback Categorization & RAG Pipeline."""
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    # Store config in context
    ctx.obj['log_level'] = log_level
    ctx.obj['config'] = config

    if config:
        click.echo(f"📁 Loading config from {config}")
        # TODO: Load additional config if provided


@cli.command()
@click.argument('transcript_file', type=click.Path(exists=True, path_type=Path))
@click.option('--transcript-id', help='Custom transcript ID')
@click.option('--output', help='Output file for results',
              type=click.Path(path_type=Path))
def process_transcript(transcript_file: Path, transcript_id: str | None,
                      output: Path | None) -> None:
    """Process a single transcript file."""
    try:
        # Validate file
        if not transcript_file.exists():
            click.echo(f"❌ Transcript file not found: {transcript_file}", err=True)
            sys.exit(1)

        if transcript_file.stat().st_size == 0:
            click.echo(f"❌ Transcript file is empty: {transcript_file}", err=True)
            sys.exit(1)

        # Read transcript
        try:
            content = transcript_file.read_text(encoding='utf-8')
        except UnicodeDecodeError as e:
            click.echo(f"❌ Cannot read transcript file (encoding issue): {e}", err=True)
            sys.exit(1)

        # Use filename as ID if not provided
        if not transcript_id:
            transcript_id = transcript_file.stem

        # Run async processing
        result = asyncio.run(_process_transcript_async(content, transcript_id))

        # Output results
        _display_processing_result(result, transcript_id)

        # Save results if requested
        if output:
            _save_results(result, output)
            click.echo(f"   💾 Results saved to: {output}")

    except KeyboardInterrupt:
        click.echo("\n❌ Processing interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error processing transcript")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


async def _process_transcript_async(content: str, transcript_id: str) -> dict[str, Any]:
    """Async wrapper for transcript processing."""
    try:
        pipeline = FeedbackPipeline()
        return await pipeline.process_transcript(content, transcript_id)
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        raise


def _display_processing_result(result: dict[str, Any], transcript_id: str) -> None:
    """Display processing results in a user-friendly format."""
    status = result.get('status', 'unknown')

    if status == 'error':
        click.echo(f"❌ Processing failed for transcript: {transcript_id}")
        click.echo(f"   Error: {result.get('error', 'Unknown error')}")
        return

    click.echo(f"✅ Processed transcript: {transcript_id}")
    click.echo(f"   📝 Feedbacks extracted: {result.get('feedbacks_extracted', 0)}")
    click.echo(f"   🎯 Matches found: {result.get('matches_found', 0)}")
    click.echo(f"   📊 Problems updated: {result.get('problems_updated', 0)}")
    click.echo(f"   📈 Status: {status}")

    # Show warnings if any
    if result.get('feedbacks_extracted', 0) == 0:
        click.echo("   ⚠️  No feedback was extracted from this transcript")
    elif result.get('matches_found', 0) == 0:
        click.echo("   ⚠️  No matches found - feedback may be too unique")


def _save_results(result: dict[str, Any], output_path: Path) -> None:
    """Save processing results to file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

@cli.command()
@click.argument('transcript_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--pattern', default='*.txt', help='File pattern to match')
@click.option('--output', help='Output file for batch results')
async def batch_process(transcript_dir: str, pattern: str, output: str) -> None:
    """Process all transcripts in a directory."""
    try:
        transcript_dir_path = Path(transcript_dir)

        # Find transcript files
        transcript_files = list(transcript_dir_path.glob(pattern))

        if not transcript_files:
            click.echo(f"❌ No files matching '{pattern}' found in {transcript_dir_path}")
            return

        click.echo(f"📂 Found {len(transcript_files)} transcript files")

        # Load transcripts
        transcripts = []
        for file_path in transcript_files:
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()

                transcripts.append({
                    "id": file_path.stem,
                    "content": content
                })
            except Exception as e:
                click.echo(f"⚠️  Warning: Could not read {file_path}: {e}")

        if not transcripts:
            click.echo("❌ No valid transcripts found")
            return

        # Process batch
        click.echo("🚀 Starting batch processing...")
        pipeline = FeedbackPipeline()
        result = await pipeline.batch_process_transcripts(transcripts)

        # Output results
        click.echo("\n✅ Batch processing completed!")
        click.echo(f"   📄 Transcripts processed: {result['total_transcripts']}")
        click.echo(f"   📝 Total feedbacks: {result['total_feedbacks']}")
        click.echo(f"   🎯 Total matches: {result['total_matches']}")
        click.echo(f"   📊 Total updates: {result['total_updates']}")
        click.echo(f"   📈 Success rate: {result['success_rate']:.1%}")
        click.echo(f"   📋 Metrics: {result['metrics_file']}")

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"   💾 Results saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise

@cli.command()
async def sync_problems() -> None:
    """Sync and embed all Notion problems."""
    try:
        click.echo("🔄 Syncing Notion problems...")

        pipeline = FeedbackPipeline()
        success = await pipeline.sync_notion_problems()

        if success:
            click.echo("✅ Problems synced successfully!")
        else:
            click.echo("❌ Failed to sync problems")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise

@cli.command()
@click.option('--feedbacks-file', help='Path to feedback logs JSON file')
@click.option('--limit', default=10, help='Number of recent feedbacks to show')
def show_feedbacks(feedbacks_file: str, limit: int) -> None:
    """Show recent extracted feedbacks."""
    try:
        from extract import load_feedback_logs

        if feedbacks_file:
            feedbacks = load_feedback_logs(feedbacks_file)
        else:
            # Find most recent feedback log
            data_dir = Path("data")
            if not data_dir.exists():
                click.echo("❌ No data directory found")
                return

            log_files = list(data_dir.glob("feedback_logs_*.json"))
            if not log_files:
                click.echo("❌ No feedback log files found")
                return

            latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
            feedbacks = load_feedback_logs(str(latest_file))

        if not feedbacks:
            click.echo("❌ No feedbacks found")
            return

        # Show recent feedbacks
        recent_feedbacks = sorted(feedbacks, key=lambda f: f.timestamp, reverse=True)[:limit]

        click.echo(f"📝 Showing {len(recent_feedbacks)} most recent feedbacks:\n")

        for i, feedback in enumerate(recent_feedbacks, 1):
            type_emoji = "🚀" if feedback.type == "feature_request" else "😓"
            click.echo(f"{i}. {type_emoji} {feedback.type.replace('_', ' ').title()}")
            click.echo(f"   📄 Transcript: {feedback.transcript_id}")
            click.echo(f"   📅 Time: {feedback.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"   🎯 Confidence: {feedback.confidence:.2f}")
            click.echo(f"   📝 Summary: {feedback.summary}")
            click.echo(f"   💬 Verbatim: \"{feedback.verbatim[:100]}{'...' if len(feedback.verbatim) > 100 else ''}\"")
            click.echo()

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

@cli.command()
def status() -> None:
    """Show pipeline status and configuration."""
    click.echo("🔧 Pipeline Configuration:")
    click.echo(f"   🤖 LLM Provider: {settings.llm_provider}")
    click.echo(f"   🧠 LLM Model: {settings.llm_model}")
    click.echo(f"   🔢 Embedding Model: {settings.embedding_model}")
    click.echo(f"   🗃️  Vector Store: {settings.vector_store}")
    click.echo(f"   📊 Notion DB: {'✅' if settings.notion_database_id else '❌'}")
    click.echo(f"   🎯 Confidence Threshold: {settings.confidence_threshold}")
    click.echo(f"   🔄 Reranking: {'✅' if settings.rerank_enabled else '❌'}")

    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        click.echo("\n📁 Data Directory:")

        transcript_count = len(list(data_dir.glob("transcripts/*.txt")))
        log_count = len(list(data_dir.glob("feedback_logs_*.json")))
        result_count = len(list(data_dir.glob("processing_results_*.json")))
        metric_count = len(list(data_dir.glob("metrics_*.json")))

        click.echo(f"   📄 Transcripts: {transcript_count}")
        click.echo(f"   📝 Feedback logs: {log_count}")
        click.echo(f"   📊 Processing results: {result_count}")
        click.echo(f"   📈 Metric files: {metric_count}")
    else:
        click.echo("\n📁 Data Directory: Not found (will be created)")

# Wrapper functions to handle async commands
def async_command(f: Any) -> Any:
    """Decorator to handle async click commands."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# Apply async wrapper to async commands
process_transcript = click.command()(async_command(process_transcript))
batch_process = click.command()(async_command(batch_process))
sync_problems = click.command()(async_command(sync_problems))

if __name__ == "__main__":
    cli()
