# Benchmark Protocol

Quorabust keeps benchmark governance separate from model choice. A protocol manifest
records what data and split roles a future result is allowed to use; it is not a model
card and it does not contain a quality number.

## Validate a manifest

Create a JSON manifest from the permitted source bytes, then run:

```bash
quorabust-validate-protocol \
  --protocol reports/quorabust-benchmark-protocol.json
```

The validator is deliberately fail-closed. A valid manifest must include:

- the dataset source reference, license/terms, source SHA-256, and a passing audit
  manifest whose source hash matches and whose `qid1`/`qid2` requirement is enabled;
- four distinct artifact references for `train`, `tuning`, `calibration`, and
  `final_holdout`, with each role limited to its declared activity;
- `question_component_holdout`, complete question-ID columns, a non-negative seed,
  an evaluation fraction, and a hashed split manifest;
- threshold selection owned by `tuning`, probability calibration owned by `calibration`,
  and final evaluation owned by `final_holdout`;
- the commit SHA, Python version, hashed dependency lock, command, and machine; and
- explicit safeguards that the roles are disjoint, the final holdout was not used for
  tuning, calibration, or model selection, and raw data is not committed.

The checked-in CI smoke fixture uses synthetic files and sets
`evidence_scope` to `protocol_only_no_quality_claim`. It proves the contract and CLI
work without turning the smoke fixture into benchmark evidence.

## Build from artifacts

Use the builder when the source, audit, role, and split artifacts are available. It hashes
the bytes on disk, verifies that the audit is passing and bound to the current source, and
records the repository commit and dependency lock before running the protocol validator:

```bash
quorabust-build-protocol \
  --config reports/quorabust-protocol-builder.json \
  --out reports/quorabust-benchmark-protocol.json
```

The builder config is JSON and must provide `protocol_name`, `dataset`, `roles`, `split`,
`decision_policy`, `dependency_lock_path`, `repository_path`, and `command`. Dataset and
role entries point to external files; `split.manifest_path` points to the exact split
manifest. Relative paths resolve from the config file, while `repository_path` is an
explicit checkout path used for the commit provenance. References and license/terms text
are recorded in the output, but raw data remains outside Git.

## Real-data handoff

For a permitted Quora release, keep the licensed source outside Git:

1. Run `quorabust-audit-data --require-question-ids` and retain its JSON output outside
   the repository with the source hash.
2. Generate the deterministic question-component split and export the exact role files,
   including the untouched final holdout. Hash each role artifact and the split manifest.
3. Record the calibration method and threshold candidates, then point each decision to
   its owning role. Never tune or calibrate against the final holdout.
4. Hash the dependency lock and record the exact training/report commands and commit.
5. Validate the completed manifest before generating a comparative model card.

This gate does not download data, inspect a license, prove row-level disjointness, or
establish model quality by itself. The audit and split artifacts remain the evidence that
must be reviewed alongside any eventual model card. See [REPORTING.md](REPORTING.md) for
the training, calibration, and report commands.

## Bind a model card

After generating a report, bind it to the protocol during release validation:

```bash
quorabust-validate-report \
  --report reports/quorabust-model-card.json \
  --protocol reports/quorabust-benchmark-protocol.json
```

Protocol-bound validation automatically requires holdout metrics, calibration diagnostics,
the reproducibility manifest, and qid-component evidence. It also checks that the report's
training source and final evaluation CSV hashes, split seed and fraction, threshold metric
and candidate grid match the protocol. The report-only validator remains available for
exploratory or customer-domain runs with an explicitly documented fallback policy.
