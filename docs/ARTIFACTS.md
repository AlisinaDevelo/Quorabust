# Artifact trust policy

Quorabust has two explicit artifact paths. The distinction is deliberate:

| Format | Supported surface | Trust boundary |
| --- | --- | --- |
| `.qmodel` | Fitted `PairFeatureBuilder` plus `XGBClassifier` baseline | Safe structured bundle; digest still required for authenticity |
| `.pkl` | All current builders, calibration wrappers, and legacy workflows | Trusted code only; loading can execute arbitrary code |

## Safe baseline export

Export a trusted baseline after training:

```bash
quorabust-export-safe \
  --model models/quorabust.pkl \
  --out models/quorabust.qmodel \
  --metadata-out models/quorabust.qmodel.meta.json
```

The `.qmodel` file is a ZIP bundle containing exactly three members: a manifest, explicit
TF-IDF vocabulary/IDF state, and the XGBoost native JSON model. Loading it reconstructs the
known feature builder and XGBoost estimator; it does not invoke Python pickle deserialization.
The export command is intentionally a trusted conversion step because its source may be a
pickle. The output preserves threshold and lineage metadata while removing local source-path
fields.

The safe path is served by the existing `load_classifier` and `/models` contracts. The model
identity includes `artifact_format`, and the existing `QUORABUST_MODEL_SHA256` pin applies to
`.qmodel` bytes before loading. The parity tests compare probabilities, threshold metadata,
and API readiness/model identity between the source baseline and the safe bundle.

## Threat model

- **Acquisition:** a digest detects accidental or malicious byte changes only when the expected
  digest comes from a trusted deployment configuration. It does not prove who produced a file;
  use a signed registry or provenance system for authenticity.
- **Registry and promotion:** inspect the sidecar, model format, source hash, code revision,
  and benchmark evidence before promotion. Never promote a `.pkl` from an untrusted registry.
- **Startup:** `.qmodel` uses structured JSON/ZIP parsing and native XGBoost JSON loading;
  `.pkl` remains restricted to controlled storage and pinned deployment images.
- **Rollback:** retain the complete artifact digest and metadata sidecar for both formats so a
  rollback names immutable bytes rather than a mutable path.
- **Resource exhaustion:** safe structured data is not automatically harmless. Apply file-size,
  CPU, memory, and timeout limits at the container/gateway boundary.

## Format decision

The current safe implementation targets the production control model first. `skops.io` is a
reasonable candidate for Python object persistence because it requires explicit trust decisions,
but the custom builder and optional transformer backends still need a compatibility matrix.
ONNX remains a future serving target: it can remove the Python runtime, but the custom pair
feature builder needs a converter and parity test before it can replace this path. Until then,
the policy is per-backend and explicit rather than pretending one format supports every model.

References: [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html),
[XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html), and
[sklearn-onnx custom converters](https://onnx.ai/sklearn-onnx/tutorial_2_new_converter.html).
