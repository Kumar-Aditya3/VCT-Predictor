# Getting Started

This guide matches the consolidated repo layout: `backend/app/*` for the API and `frontend/app/*` for the dashboard.

## 1. Install dependencies

### Backend
```bash
python -m pip install -r backend/requirements.txt
```

### Frontend
```bash
cd frontend
npm install
cd ..
```

## 2. Run the backend

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Verify it:

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/scope
```

## 3. Run the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

The dashboard tries the backend first and falls back to local bootstrap data if the API is unavailable.
Additional pages:
- `http://localhost:3000/analytics`
- `http://localhost:3000/validation`

## 4. Run the weekly pipeline

```bash
cd backend
python -m scripts.weekly_update
python -m scripts.validate_vlr_ground_truth
```

Artifacts are written to `artifacts/`.

## 5. Run tests

```bash
python -m unittest discover -s backend/tests
```

## API Reference

Base URL: `http://localhost:8000/api/v1`

- `GET /health`
- `GET /scope`
- `POST /predict`
- `GET /predictions/upcoming`
- `GET /model/performance`
- `GET /data/validation`
- `POST /pipeline/weekly`

Example prediction request:

```bash
curl -X POST http://localhost:8000/api/v1/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"fixtures\":[{\"match_id\":\"sample-1\",\"region\":\"Pacific\",\"event_name\":\"Masters\",\"team_a\":\"Paper Rex\",\"team_b\":\"Gen.G\",\"match_date\":\"2026-04-08\",\"best_of\":3}]}"
```

## Notes

- Prediction mode switches to `trained_ml` after a successful weekly training run.
- Weekly validation is backed by VLR list-page and match-detail ingestion, SQLite-backed canonical data, audit/exclusion tracking, rolling model search, and calibrated match/map outputs.
- Canonical local data is stored in `data/processed/vct_tier1.sqlite3` unless `SQLITE_DB_PATH` is overridden.
- The old root-level API and Pages Router scaffold are no longer part of the supported workflow.
