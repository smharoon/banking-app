from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select

from app.database import (
    COLUMNS, batches, init_db, ingest_dataframe, load_csv,
    records, sha256_file, update_batch,
)
from app.report import build_report

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example_bank_marketing.csv"


def test_single_record_database_and_report(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    df = load_csv(SAMPLE)
    assert list(df.columns) == COLUMNS
    batch_id, duplicate = ingest_dataframe(engine, df, sha256_file(SAMPLE), SAMPLE.name)
    assert batch_id == 1 and not duplicate
    output, quality = build_report(df, batch_id, SAMPLE.name, tmp_path / "reports")
    assert output.exists() and "DataOps Report" in output.read_text()
    assert quality["injected_age_nulls"] == 1
    update_batch(engine, batch_id, "completed", report_path=output, quality_metrics=quality)
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(records)).scalar_one() == 1
        row = connection.execute(select(batches)).one()
        assert row.status == "completed"
    _, duplicate = ingest_dataframe(engine, df, sha256_file(SAMPLE), SAMPLE.name)
    assert duplicate


def test_bad_header_is_rejected(tmp_path):
    f = tmp_path / "invalid.csv"
    f.write_text("age;wrong\n23;x\n")
    with pytest.raises(ValueError, match="Incorrect CSV headers"):
        load_csv(f)


def test_report_with_two_classes(tmp_path):
    base = load_csv(SAMPLE)
    df = pd.concat([base] * 30, ignore_index=True)
    df.loc[:11, "y"] = "yes"
    df.loc[:14, "job"] = "technician"
    df.loc[:14, "age"] = 35
    output, quality = build_report(df, 20, "synthetic-test.csv", tmp_path / "reports")
    assert output.exists() and output.stat().st_size > 10000
    assert "Exploratory Random Forest feature importance" in output.read_text()
    assert quality["injected_age_nulls"] >= 1
