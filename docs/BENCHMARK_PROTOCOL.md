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

Start with the path-light template at
[`examples/protocol-builder.config.example.json`](../examples/protocol-builder.config.example.json).
Replace every `/absolute/path` and `REPLACE_WITH_...` value after reviewing the dataset
license and terms; the template contains no benchmark data.

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

Config policy is checked before any artifact is opened. Seeds must be non-negative integers;
fractions and threshold candidates must be finite and strictly between zero and one; IDs
must include `qid1` and `qid2` without duplicates; and the threshold grid must not repeat a
value. ID columns and threshold candidates are sorted in the emitted manifest, so equivalent
unordered configs produce the same policy ordering.

## Real-data handoff

For a permitted Quora release, keep the licensed source outside Git:

1. Run the freezer against the permitted source. It always performs the strict audit first,
   refuses existing output files, and writes all four role CSVs outside the repository:

   ```bash
   quorabust-freeze-protocol \
     --csv /external/quora/question_pairs.csv \
     --out-dir /external/quora/quorabust-roles \
     --audit-out /external/quora/quorabust-audit.json \
     --split-out /external/quora/quorabust-split.json \
     --seed 42 \
     --tuning-fraction 0.1 \
     --calibration-fraction 0.1 \
     --final-holdout-fraction 0.1
   ```

   The command writes `train.csv`, `tuning.csv`, `calibration.csv`, and
   `final_holdout.csv` from whole question components. Its split manifest records source,
   audit, role, and provenance hashes plus per-role label counts; repeat the command in a
   fresh output directory to reproduce the same bytes from the same source bytes and seed.
   Each role must contain both label classes so calibration and final evaluation fail before
   artifact generation rather than later in the release pipeline.
2. Review the passing audit and role counts, including the untouched final holdout. The
   freezer fails closed for missing or incomplete IDs, invalid labels, insufficient
   components, invalid fractions, and output collisions.
3. Train only on the `train.csv` role and pass `tuning.csv` as `quorabust-train --eval-csv`.
   The command verifies component disjointness and records the train and tuning hashes.
   Never train on the full source after roles have been frozen.
4. Record the calibration method and threshold candidates, then point each decision to
   its owning role. If the threshold metric is `expected_cost`, also record positive
   `false_positive_cost` and `false_negative_cost` values in `decision_policy`; the
   report validator binds the artifact's persisted cost matrix to those values. Never
   tune or calibrate against the final holdout.
5. Hash the dependency lock and record the exact training/report commands and commit.
6. Validate the completed manifest before generating a comparative model card. Protocol-bound
   validation checks the train, tuning, calibration, and final-holdout role hashes rather
   than treating the raw source hash as the training artifact.

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
and candidate grid match the protocol. For `expected_cost`, it also checks both cost
values. The report-only validator remains available for
exploratory or customer-domain runs with an explicitly documented fallback policy.
