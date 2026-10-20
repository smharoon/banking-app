"""Database schema, CSV validation and idempotent batch ingestion."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, MetaData, String,
    Table, Text, UniqueConstraint, create_engine, delete, insert, select, update,
)

COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y",
]
NUMERIC = ["age", "duration", "campaign", "pdays", "previous",
           "emp.var.rate", "cons.price.idx", "cons.conf.idx",
           "euribor3m", "nr.employed"]
CATEGORICAL = [c for c in COLUMNS if c not in NUMERIC]

metadata = MetaData()
batches = Table(
    "data_batches", metadata,
    Column("batch_id", Integer, primary_key=True, autoincrement=True),
    Column("filename", String(255), nullable=False),
    Column("file_checksum", String(64), nullable=False, unique=True),
    Column("status", String(20), nullable=False),
    Column("row_count", Integer),
    Column("quality_metrics", Text),
    Column("report_path", Text),
    Column("error_message", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
)
records = Table(
    "bank_marketing_records", metadata,
    Column("record_id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", Integer, ForeignKey("data_batches.batch_id"), nullable=False),
    Column("source_row", Integer, nullable=False),
    *[Column(c, Float if c in NUMERIC else String(100)) for c in COLUMNS],
    UniqueConstraint("batch_id", "source_row", name="uq_batch_row"),
)


def db_engine():
    return create_engine(
        os.getenv("DATABASE_URL", "sqlite:///bank_demo.db"),
        pool_pre_ping=True,
    )


def init_db(engine):
    metadata.create_all(engine)


def load_csv(path: Path) -> pd.DataFrame:
    # UCI Bank Marketing uses semicolon separators and may include a UTF-8 BOM.
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()
    if list(df.columns) != COLUMNS:
        raise ValueError(f"Incorrect CSV headers. Expected, in order: {COLUMNS}; got: {list(df.columns)}")
    if df.empty:
        raise ValueError("CSV is empty")
    for col in NUMERIC:
        # Reject malformed numeric values, but allow genuinely empty cells.
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df["y"].isna().any() or not df["y"].isin(["yes", "no"]).all():
        raise ValueError("Target y must contain only yes or no")
    return df


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_dataframe(engine, df: pd.DataFrame, checksum: str, filename: str) -> tuple[int, bool]:
    """Store original records, return (batch_id, is_duplicate_complete).

    A failed/retried checksum is replaced transactionally, so rows are never
    appended twice to an existing batch.
    """
    with engine.begin() as connection:
        row = connection.execute(
            select(batches.c.batch_id, batches.c.status)
            .where(batches.c.file_checksum == checksum)
        ).first()
        if row and row.status == "completed":
            return int(row.batch_id), True
        if row:
            batch_id = int(row.batch_id)
            connection.execute(delete(records).where(records.c.batch_id == batch_id))
            connection.execute(
                update(batches).where(batches.c.batch_id == batch_id).values(
                    filename=filename, status="processing", row_count=len(df),
                    error_message=None, report_path=None,
                )
            )
        else:
            result = connection.execute(insert(batches).values(
                filename=filename, file_checksum=checksum, status="processing",
                row_count=len(df), created_at=datetime.now(timezone.utc),
            ))
            batch_id = int(result.inserted_primary_key[0])

        # Preserve raw validated data for later MLOps training/evaluation.
        # Pandas NA must be converted into actual SQL NULL rather than NaN.
        raw = df.astype(object).where(pd.notna(df), None)
        for begin in range(0, len(raw), 1000):
            chunk = raw.iloc[begin:begin + 1000]
            payload = []
            for i, values in enumerate(chunk.itertuples(index=False, name=None), start=begin + 1):
                record = dict(zip(COLUMNS, values))
                record.update(batch_id=batch_id, source_row=i)
                payload.append(record)
            connection.execute(insert(records), payload)
    return batch_id, False


def update_batch(engine, batch_id: int, status: str, *, report_path=None,
                 quality_metrics=None, error_message=None) -> None:
    fields = {"status": status, "error_message": error_message}
    if status == "completed":
        fields["completed_at"] = datetime.now(timezone.utc)
    if report_path is not None:
        fields["report_path"] = str(report_path)
    if quality_metrics is not None:
        fields["quality_metrics"] = json.dumps(quality_metrics)
    with engine.begin() as connection:
        connection.execute(update(batches).where(batches.c.batch_id == batch_id).values(**fields))
