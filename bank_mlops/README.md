# Bank Marketing MLOps — separate from DataOps

Two Prefect flows:
- **train_model**: manual/ad hoc; combine all `train_model/incoming/*.csv`, insert source rows in `mlops.training_records`, split 70/30, train Logistic Regression and Random Forest, evaluate both, register winner with the MLflow `champion` alias, archive successful files.
- **evaluate_model**: every 120 seconds; for each `evaluate_model/incoming/*.csv`, insert source rows in `mlops.evaluation_records`, load MLflow champion, calculate file metrics and row-level predictions, log MLflow, archive successful files.

**DataOps tables are not referenced or modified.** Only the PostgreSQL *server* and Prefect *server* are shared.

## 1. Prerequisites

Your existing `infrastructure/` Compose project must have running PostgreSQL and Prefect containers on the `bank-network` Docker network, accessible using Docker aliases `bank-postgres` and `bank-prefect`. Add MLflow to the infrastructure Compose file using `infra_mlflow_snippet.yaml` as a reference. This provides alias `bank-mlflow` and port 5000. MLflow artifacts and metadata must be persisted.

## 2. Start

```bash
cp .env.example .env
# EDIT .env with the actual bank database password and settings
# Run after infrastructure + MLflow are running:
docker compose up -d --build
```

Verify:
```bash
docker compose ps
docker compose logs --tail=60 mlops-train
docker compose logs --tail=60 mlops-evaluate
```

## 3. Training (manual)

Put labeled semicolon-delimited UCI `bank-additional-full.csv` format CSVs in `train_model/incoming/`. Each file must contain the 21 original headers including `y`. All NEW files are combined into one training dataset per manual flow execution. Trigger in Prefect UI (`bank-marketing-train-model/manual-training`), or run:

```bash
docker compose exec mlops-train python -m app.train_flow --once
```

Data stored in `mlops.files`, `mlops.training_records`, and `mlops.training_runs`. Training CSVs are moved to `train_model/processed/` after success. The champion alias points to the best model of that run (highest positive-class F1). The training test set is a stratified 30% holdout.

## 4. Evaluation (scheduled or manual)

Put labeled CSVs in `evaluate_model/incoming/`, with **observations not used in model training**. Each file is independently evaluated using saved champion model *without retraining*.

```bash
docker compose exec mlops-evaluate python -m app.evaluate_flow --once
```

Rows and results are stored under `mlops.evaluation_records`, `mlops.evaluation_metrics`, and `mlops.predictions`. CSV containing row-level predictions is written to `outputs/`. Upon success the CSV moves to `evaluate_model/processed/`.

## 5. Unlabeled predictions

```bash
docker compose exec mlops-evaluate python -m app.predict --input /app/evaluate_model/incoming/unlabeled.csv --output /app/outputs/predicted.csv
```

Unlabeled prediction CSV must contain all predictive feature columns but may omit `duration` and `y`. Standalone predictions don't create an evaluation metrics run.

## 6. Dashboards

- Prefect: http://localhost:4200
- MLflow: http://localhost:5000 (when tunneled locally or bound to loopback)

## Important behavior and limitations

- DataOps is *not required* to run for MLOps to operate; infrastructure PostgreSQL, Prefect, and MLflow must be available.
- Both training and evaluation store **original validated data**, no synthetic missing-value injection. Imputation/encoding and scaling are fitted on the 70% training portion only, and saved in the registered pipeline.
- `duration` is excluded from model inputs due to pre-contact information leakage. `pdays=999` is retained as a numeric sentinel in this simple version; it is not treated as literal days.
- `train_model` considers **new pending files** per run, not previously processed files. Reusing the same file contents is deduplicated by SHA-256; to do full retraining combine the original files in a new deliberate run strategy.
- This is an assignment/demo-scale implementation. Two flows can run in separate containers. Concurrent/manual repeat triggers should be avoided; serialize train and evaluation execution while upgrading to production idempotence.
- Never run `docker compose down -v` on your infrastructure if you want to retain its database and MLflow volumes.
- For security, keep PostgreSQL and MLflow private or behind authentication and a reverse proxy.
