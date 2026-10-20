# Bank Marketing DataOps — CCZG506 Assignment I

A **minimal, runnable batch DataOps pipeline**: Prefect 3 + Python + PostgreSQL + Docker Compose.

**Flow:** Every 120 seconds → scan `data/incoming` → claim CSV → validate headers → ingest original rows into PostgreSQL → run preprocessing/EDA on an in-memory copy → write an independent HTML report → archive CSV. Prefect logs scheduled runs and failures. No separate Prefect worker is needed: `flow.serve()` registers and executes this small deployment.

## Requirements

- Docker Engine / Docker Desktop with `docker compose`.
- Enough free memory for Prefect Server, PostgreSQL, scikit-learn and Python (4 GB available to Docker is a good starting point).
- Use CSV files with the **semicolon (`;`) delimiter** and the exact 21 UCI `bank-additional-full.csv` headers.

## Quick start

```bash
# Run from the bank_dataops directory
cp .env.example .env
# Edit .env and choose a password BEFORE the first startup.
docker compose up -d --build

# Copy the one-row sample to demonstrate ingestion (Linux/macOS)
cp examples/example_bank_marketing.csv data/incoming/

# See runner logs
docker compose logs -f dataops
```

For Windows PowerShell use `Copy-Item .env.example .env` and `Copy-Item examples/example_bank_marketing.csv data/incoming/`.

Open **http://localhost:4200** for Prefect. You should see the flow `bank-marketing-dataops` and deployment `every-2-minutes`. An input file is discovered at the next scheduled run (up to ~2 minutes after the schedule is active). To process files immediately:

```bash
docker compose exec dataops python -m app.flow --once
```

Inspect the generated report at `reports/batch_0001_report.html` (open it in your browser), the source file at `data/archive/`, and the Prefect flow run logs. **The one-row example tests the plumbing only.** To demonstrate useful correlations, class associations and feature importance, copy the complete 41,188-row `bank-additional-full.csv` into `data/incoming` instead.

### Check stored records

```bash
docker compose exec postgres psql -U bank -d bank_data -c 'SELECT batch_id, filename, status, row_count, report_path FROM data_batches ORDER BY batch_id;'

docker compose exec postgres psql -U bank -d bank_data -c 'SELECT batch_id, COUNT(*) FROM bank_marketing_records GROUP BY batch_id ORDER BY batch_id;'
```

Optional Prefect API test (once the server is running):

```bash
curl -X POST http://localhost:4200/api/flows/filter -H 'Content-Type: application/json' -d '{"limit":5}'
```

### Folder conventions

- `data/incoming`: users drop new `.csv` files here.
- `data/processing`: files are atomically claimed by a flow run.
- `data/archive`: successful files and exact-content duplicates.
- `data/failed`: failed file for inspection; copy it back into `incoming` to retry after fixing the problem.
- `reports`: standalone HTML report for each successfully processed unique checksum.

Duplicates are identified by SHA-256 hash and skipped after a batch is marked completed. On an explicit retry of a previously failed checksum, the batch rows are replaced transactionally. Restart recovery from files abandoned in `processing` due to abrupt machine/container termination is **not automated**; manually move those files back into `incoming` to retry.

### Database contract for the future MLOps pipeline

`data_batches` stores `batch_id`, filename, hash, status, row count, quality metrics (JSON text), timestamps and report path. `bank_marketing_records` stores `record_id`, `batch_id`, `source_row`, and **all 21 original UCI columns** (including `y`). These are raw validated observations, not transformed data. A future MLOps pipeline can query `WHERE batch_id IN (...)` and perform its own training-only imputation/encoding/scaling.

### EDA implemented

- Shape, data types, numeric summary statistics, row duplicates.
- Missing-value checks, `unknown` counts, 5% induced nulls in the `age` column on a working copy, median imputation, before/after count.
- Numeric z-score normalization, one-hot encoding summary, age binning.
- Pearson correlation heatmap; Cramér's V for categorical predictors vs target.
- Age, job, target plots; subscription rate vs job.
- Exploratory Random Forest feature importance (only when there are >=20 rows and two target classes, **excluding duration**). This is not the MLOps training/evaluation flow.

`pdays = 999` means no previous contact; it is treated specially when building the feature-importance example. Call duration is excluded from that example because it is not available before calling.

### Errors / limitations

- The sample is **one data row only**. Correlation coefficients and feature importance require more diverse data, so these are shown conditionally in reports.
- Scheduled runs with no new files finish successfully and log `No new CSV files`.
- Files are accepted only when their headers match the original dataset exactly and `y` is `yes` / `no`.
- Automatic preprocessing is for EDA/reporting; transformed rows are **not** stored in PostgreSQL to avoid data leakage into MLOps.
- `.env` credentials are for this course demo, not an internet-facing production system. PostgreSQL is not exposed to the host and Prefect binds **only to localhost** by default; for a cloud VM, use an SSH tunnel to access its UI, e.g. `ssh -L 4200:127.0.0.1:4200 user@your-vm`.
- This sample is a **single-VM Docker deployment** suitable for learning and demo use; it does not include HA, alerting or managed cloud storage.

### Stop

```bash
docker compose down
```

`docker compose down -v` additionally **deletes persistent databases and Prefect history**; do not run it unless intentionally resetting the project.

## Academic dataset citation

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. https://doi.org/10.1016/j.dss.2014.03.001
