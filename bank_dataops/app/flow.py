"""Prefect scheduled folder polling / ingestion / EDA flow."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from prefect import flow, get_run_logger, task

from app.database import db_engine, init_db, ingest_dataframe, load_csv, sha256_file, update_batch
from app.report import build_report

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", "reports"))


def ensure_dirs():
    for name in ("incoming", "processing", "archive", "failed"):
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def move_to(folder: str, file: Path, source_name: str) -> Path:
    dest = DATA_DIR / folder / source_name
    if dest.exists():
        dest = DATA_DIR / folder / f"{uuid.uuid4().hex[:8]}_{source_name}"
    shutil.move(str(file), str(dest))
    return dest


@task(name="Discover CSV files")
def discover_files() -> list[str]:
    # A file is claimed by an atomic rename before it can be ingested.
    return [str(path) for path in sorted((DATA_DIR / "incoming").glob("*.csv"))]


@task(name="Validate and ingest CSV")
def ingest_file(file: str, original_name: str):
    path = Path(file)
    checksum = sha256_file(path)
    df = load_csv(path)
    engine = db_engine()
    batch_id, duplicate = ingest_dataframe(engine, df, checksum, original_name)
    return batch_id, duplicate, df


@task(name="Preprocess, EDA and HTML report")
def make_report(df, batch_id: int, filename: str):
    return build_report(df, batch_id, filename, REPORT_DIR)


@flow(name="bank-marketing-dataops", log_prints=True)
def dataops_pipeline():
    logger = get_run_logger()
    ensure_dirs()
    engine = db_engine()
    init_db(engine)
    discovered = discover_files()
    if not discovered:
        logger.info("No new CSV files. Flow completed successfully.")
        return
    logger.info("Discovered %d new CSV file(s)", len(discovered))
    failed_count = 0
    for filename in discovered:
        source = Path(filename)
        source_name = source.name
        # Move out of incoming first to avoid being picked by another run.
        processing = DATA_DIR / "processing" / f"{uuid.uuid4().hex}_{source_name}"
        try:
            source.rename(processing)
        except FileNotFoundError:
            logger.info("Already claimed by another run: %s", source_name)
            continue
        batch_id = None
        try:
            batch_id, duplicate, df = ingest_file(str(processing), source_name)
            if duplicate:
                move_to("archive", processing, f"duplicate_{source_name}")
                logger.info("Skipping duplicate file %s (existing batch %d)", source_name, batch_id)
                continue
            logger.info("Loaded batch %d: %s (%d rows)", batch_id, source_name, len(df))
            report_path, quality = make_report(df, batch_id, source_name)
            archive_path = move_to("archive", processing, source_name)
            update_batch(engine, batch_id, "completed", report_path=report_path, quality_metrics=quality)
            logger.info("Batch %d complete. Report: %s. Archived: %s", batch_id, report_path, archive_path)
        except Exception as exc:
            failed_count += 1
            if batch_id is not None:
                try:
                    update_batch(engine, batch_id, "failed", error_message=str(exc))
                except Exception:
                    logger.exception("Could not update database status for batch %d", batch_id)
            if processing.exists():
                target = move_to("failed", processing, source_name)
                logger.error("File %s failed: %s. Saved at %s", source_name, exc, target)
            else:
                logger.exception("Failure after moving file %s: %s", source_name, exc)
    # Mark the flow as failed if any input failed, without preventing other
    # independent files from being processed on this run.
    if failed_count:
        raise RuntimeError(f"{failed_count} file(s) failed; check logs and data/failed")


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        dataops_pipeline()
    else:
        # 'serve' both registers the deployment and executes scheduled runs.
        # No separate Prefect worker or queue is needed for this small project.
        dataops_pipeline.serve(name="every-2-minutes", interval=120,
                               pause_on_shutdown=False, limit=1)
