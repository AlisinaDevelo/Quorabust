# Demo API snapshots

Generated from `examples/smoke_pairs.csv` with `quorabust-demo-assets`.
These files demonstrate the serving contract and are not model-quality claims.

- `predict-request.json`: sample batch scoring payload.
- `predict-response.json`: response shape with a request threshold override and feature values.
- `models-response.json`: safe public model metadata returned by `/models`.
- `openapi-excerpt.json`: focused OpenAPI slice for `/predict`, `/models`, and `/metrics`.

Regenerate:

```bash
quorabust-demo-assets --out docs/demo-assets
```

CI freshness check:

```bash
quorabust-demo-assets --out docs/demo-assets --check
```
