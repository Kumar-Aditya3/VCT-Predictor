# VCT Tier 1 Predictor

VCT Tier 1 prediction platform built around a single FastAPI backend and Next.js dashboard.

Current core capabilities:
- Match winner, map outcome, and player kill/death projections are trained from VLR-backed Tier 1 history.
- Ingestion follows Tier 1 VLR listing pages into match-detail pages to extract maps, player stat lines, and agent picks when available.
- SQLite is the local source of truth for canonical matches, maps, player map stats, aliases, audit issues, exclusions, and pipeline runs.
- Tier 1 scope is enforced for Americas, EMEA, Pacific, and China, with International handled for global Masters/Champions events.
- Weekly automation ingests fresh VLR data, validates detail integrity, retrains the published bundle with rolling backtests and calibration, and writes structured prediction and validation artifacts.

## Supported Architecture

### Backend
- Canonical entrypoint: `backend/app/main.py`
- Run locally:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

- API base: `http://localhost:8000/api/v1`

Supported routes:
- `GET /health`
- `GET /scope`
- `POST /predict`
- `GET /predictions/upcoming`
- `GET /model/performance`
- `GET /data/validation`
- `POST /pipeline/weekly`

### Frontend
- Canonical entrypoint: `frontend/app/*`
- Stack: Next.js 16 + React 19
- Run locally:

```bash
cd frontend
npm install
npm run dev
```

- Dashboard: `http://localhost:3000`
- The home page reads from `GET /api/v1/predictions/upcoming` and falls back to local bootstrap data if the API is unavailable.
- Additional pages:
  - `/analytics`
  - `/validation`

## Weekly Pipeline

Run the supported weekly flow from the backend:

```bash
cd backend
python -m scripts.weekly_update
python -m scripts.validate_vlr_ground_truth
```

What it does:
- fetches completed and upcoming matches from VLR list pages
- follows Tier 1 matches into VLR detail pages for maps, player stats, and agent usage
- filters them to Tier 1 scope
- validates scraped detail rows before training and records exclusions/audit issues
- stores canonical history and audit metadata in SQLite
- trains persisted match, map, and player projection models from historical results using broad model search
- calibrates match and map probabilities and publishes candidate leaderboards
- generates prediction snapshots for upcoming fixtures
- computes validation metrics for match winners, maps, and player kill/death projections
- writes raw source snapshots to `data/raw/`
- writes canonical training data to `data/processed/`
- writes model artifacts to `artifacts/models/`
- writes a structured artifact to `artifacts/pipeline_run_YYYYMMDD_HHMMSS.json`

## Testing

Backend tests:

```bash
python -m unittest discover -s backend/tests
```

## Project Layout

```text
vfl/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   └── services/
│   ├── scripts/
│   └── tests/
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
├── ops/
│   └── tasks/
├── data/
└── artifacts/
```

## Current Limitations

- VLR markup is unofficial and can shift, so scraper maintenance is part of normal operation.
- Agent coverage depends on what is exposed on the underlying VLR detail pages for a given match.
- Player projections currently focus on kills and deaths only.
- Heavy offline search makes weekly training materially slower than the original lightweight pipeline.
- Frontend tests are not configured in this workspace.

## Windows Automation

`setup.bat` installs dependencies and registers a weekly scheduled task that runs:

```text
ops\tasks\weekly_refresh.ps1
```

That task executes the supported backend scripts from the canonical app stack.
