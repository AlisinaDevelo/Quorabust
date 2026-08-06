# Hosted demo checklist

Use a hosted demo only as a recruiter-facing inspection surface for the API contract. The
checked-in smoke model is not a benchmark and must be labeled as a demo model anywhere it
is exposed.

## Minimum demo contract

- Public HTTPS URL for FastAPI docs at `/docs`.
- `/health` returns `{"status":"ok"}`.
- `/ready` returns `{"status":"ready"}` after the smoke artifact is loaded.
- `/models` shows safe metadata without local paths.
- `/predict?explain=true&threshold=0.9` accepts the sample request from
  [DEMO.md](DEMO.md) and returns `proba_duplicate`, `is_duplicate`,
  `decision_threshold`, `variant`, and feature values.
- Responses include `X-Request-ID`; use it when sharing a support/debugging example.
- A visible note near the link says: **demo smoke model, not production quality**.

## Build inputs

Create the smoke artifact before deploying or during a release build:

```bash
mkdir -p models
quorabust-train \
  --csv examples/smoke_pairs.csv \
  --out models/quorabust-smoke.pkl \
  --metadata-out models/quorabust-smoke.meta.json \
  --eval-fraction 0 \
  --seed 7
```

Use the container image or install with serving dependencies:

```bash
pip install ".[serve]"
quorabust-serve --host 0.0.0.0 --port "${PORT:-8000}" --model models/quorabust-smoke.pkl
```

## Runtime configuration

Required:

- `QUORABUST_MODEL_PATH=models/quorabust-smoke.pkl` if not passing `--model`.
- A platform-provided `PORT`, or explicitly run on `8000`.

Optional:

- `QUORABUST_DECISION_THRESHOLD=0.5` for a default decision cutoff.
- `QUORABUST_MODEL_B=...` only when demonstrating A/B routing.
- `QUORABUST_MODEL_SHA256=...` to pin the deployed smoke artifact bytes.
- `QUORABUST_MODEL_B_SHA256=...` to pin an A/B artifact when used.
- `QUORABUST_API_KEY=...` to require `X-Quorabust-API-Key` on scoring and model metadata.
- `QUORABUST_MAX_BATCH_SIZE=32` (or another small value) to bound public demo work.
- `QUORABUST_MAX_TEXT_LENGTH=8192` (or a smaller value) to reject oversized question strings
  before feature construction.

## Safety boundaries

- Do not expose private or customer-trained artifacts from this demo.
- Do not accept untrusted uploaded pickle artifacts.
- Pin the smoke artifact with `QUORABUST_MODEL_SHA256`; a digest mismatch should prevent
  startup rather than serve an unreviewed file.
- Use `QUORABUST_API_KEY` and a small batch cap for a basic public-demo boundary; keep TLS,
  rate limiting, quotas, and key rotation at the host or gateway.
- Keep the per-question text cap enabled; rejected text is neither logged nor echoed in the
  stable 413 error response.
- Keep request logs privacy-safe and ship only the structured operational event fields;
  never expose question text or API keys through logs.
- Disable or protect the demo when it is no longer useful; the goal is inspection, not
  production traffic.

## Portfolio capture

Capture these for the portfolio or README:

- `/docs` screenshot showing the scoring endpoint.
- `/models` screenshot showing safe metadata.
- One formatted `/predict?explain=true&threshold=0.9` response.
- Optional Grafana screenshot from the local Compose stack in [DEMO.md](DEMO.md).
