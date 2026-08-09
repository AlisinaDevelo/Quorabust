# Real Benchmark Result

This page records one permitted Quora Question Pairs evaluation run. It is evidence for
review, not a claim of state-of-the-art quality. The raw dataset and frozen role CSVs stay
outside Git.

## Corrected Strict v2 Protocol

The earlier role set below is retained for audit history. This is the current corrected
comparison: three rows with missing or blank question text were removed from a permitted
external source copy, then the strict freezer was rerun under the current release code.

- Source: `external://quora/quorabust-qqp-strict-v1.csv`
- Corrected source SHA-256: `9bbef5123fc764a2b2232752edacea21a0d1e5c303c48d8340263ed78d829d8c`
- Source-cleaning manifest SHA-256: `2d07609cd853d354a760b490dedcbc7aa1fe3cc5a2c5fd20878ed9d07a65deb4`
- Corrected source rows: 404,348; final holdout rows: 40,437; positive rate: 0.3685
- Split: question-component holdout, seed `42`, four roles at 70/10/10/10 percent.
- Train role SHA-256: `ee76ccb99b5784a4f9be3cd66e393369e9813167f2d52349237c61661b116ab3`
- Tuning role SHA-256: `ba982060fb37884bdeae2e1238fcad460ef2465387844967180fed5447e34c69`
- Calibration role SHA-256: `e41eec62728934614de333b957d2e2ebdadfc0102119df830b96cb832c333f37`
- Final holdout SHA-256: `cedad10a2d5af66cb88fbdc8cfd9edc201234d0aa8e319e3dad9b2ab73181744`
- Strict audit SHA-256: `2be55ea1d284d4551233d6a97d9c73a8812a1ada545f84d19a2e19d2308d6885`
- Split manifest SHA-256: `260435a10d8fad0ee03a8c00c2e083fe206165729f4027ed4b7490201e577714`
- Protocol SHA-256: `eca7596aefa3ee69a0b7a33db4b3f80a239723e3453d5b2d0e034e300777b6c4`
- Evaluation/report code: commit `e349581fc1b5e531c22897b8039936bc78ff8f11`
- Audit policy: `require_question_ids: true`, `require_question_text: true`

The source-cleaning manifest records only the original/output hashes, row counts, the
drop policy, and row fingerprints; raw question text remains outside Git. The protocol,
all role audits, and the protocol-bound TF-IDF report validator passed.

## Corrected Strict v2 Pair Classification

All three rows use the same untouched final holdout, tuning threshold candidates, isotonic
calibration role, and strict role hashes. The TF-IDF row is a serialized Quorabust safe
artifact and passed protocol-bound report validation. The embedding row is a trusted
pickle artifact; the cross-encoder row is direct pretrained scoring evidence only. Neither
transformer-backed row is currently a safe serialized Quorabust artifact.

| candidate | threshold | ROC-AUC | PR-AUC | log loss | Brier | ECE | precision | recall | F1 | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + XGBoost + isotonic | 0.30 | 0.7896 | 0.6243 | 0.5111 | 0.1761 | 0.0063 | 0.5306 | 0.9130 | 0.6711 | 0.6702 |
| Sentence-transformer embedding + XGBoost + isotonic | 0.40 | 0.8834 | 0.7799 | 0.4023 | 0.1341 | 0.0075 | 0.6785 | 0.8371 | 0.7495 | 0.7938 |
| Direct Quora cross-encoder + isotonic | 0.40 | 0.9731 | 0.9472 | 0.2007 | 0.0596 | 0.0034 | 0.8731 | 0.9159 | 0.8940 | 0.9199 |

### Threshold Sweep

Every cell below is `F1 (FP/FN)` on the same final holdout. A `*` marks the threshold
selected on the tuning role by F1; the sweep itself does not tune on the final holdout.

| threshold | TF-IDF + XGBoost | embedding + XGBoost | direct cross-encoder |
| ---: | ---: | ---: | ---: |
| 0.20 | 0.6599 (14,084/629) | 0.7271 (10,162/584) | 0.8753 (3,529/559) |
| 0.30* | 0.6711 (12,039/1,296) | 0.7463 (8,068/1,229) | 0.8902 (2,550/902) |
| 0.40* | 0.6695 (10,112/2,316) | 0.7495 (5,912/2,428) | 0.8940 (1,984/1,254) |
| 0.50 | 0.6306 (6,768/4,923) | 0.7284 (4,303/3,900) | 0.8920 (1,681/1,551) |
| 0.60 | 0.3182 (1,415/11,815) | 0.7072 (3,386/4,897) | 0.8866 (1,384/1,934) |
| 0.70 | 0.1381 (378/13,769) | 0.5854 (1,660/8,048) | 0.8759 (1,086/2,443) |
| 0.80 | 0.0343 (53/14,641) | 0.3394 (390/11,777) | 0.8606 (892/2,972) |

### Operating Point And Cost Review

The selected operating points have the following final-holdout confusion counts. The last
column is an equal-unit diagnostic `FP + FN` only; no application-specific false-positive
or false-negative cost weights were supplied, so it is not a deployment cost policy.

| candidate | threshold | TN | FP | FN | TP | equal-unit errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + XGBoost | 0.30 | 13,496 | 12,039 | 1,296 | 13,606 | 13,335 |
| embedding + XGBoost | 0.40 | 19,623 | 5,912 | 2,428 | 12,474 | 8,340 |
| direct cross-encoder | 0.40 | 23,551 | 1,984 | 1,254 | 13,648 | 3,238 |

### Promotion Decision

- **Promote as the current supported control:** TF-IDF + XGBoost + isotonic remains the only
  candidate with a safe `.qmodel` artifact, small artifact size, and high warm throughput.
- **Keep as the quality candidate:** the direct cross-encoder is materially stronger on this
  holdout, but its direct score run has no serialized Quorabust artifact and has roughly 1.6 GB
  peak RSS. It is not promoted from this benchmark alone.
- **Keep as the intermediate candidate:** embedding improves substantially over TF-IDF at a
  lower cost than the cross-encoder, but its trusted pickle is large and not safe for untrusted
  distribution.
- **Decision:** do not replace the supported production control yet. Package and parity-test a
  safe transformer artifact, then repeat cold-start/load and representative domain checks before
  selecting a transformer for promotion.

The embedding candidate used `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its trusted pickle was 173,917,653 bytes,
loaded in 4,216 ms, scored the final holdout at 1,057 pairs/second, and reached a process
peak RSS of 1,287,618,560 bytes on this machine. It is a useful middle point between the
cheap lexical control and the cross-encoder, but safe transformer artifact packaging and
deployment load testing remain open.

The cross-encoder used `cross-encoder/quora-distilroberta-base` at revision
`f62e7a4b20b97195c2868e53ec59126df5eac743`, `max_length=128`, batch size `128`, and explicit
CPU execution. Model load was 1,431 ms. Scoring throughput was 173.3, 163.1, and 183.7
pairs/second for tuning, calibration, and final holdout respectively; peak RSS was
approximately 1.6 GB. This is the central enterprise tradeoff: the model quality jump is
substantial, but the serving cost and artifact packaging still need engineering before
promotion.

For the same final holdout, a warm single-process pass through the protocol-bound TF-IDF
safe artifact scored 40,437 pairs in 1,160.7 ms, or 34,839 pairs/second. The artifact is
354,745 bytes and the measured process peak RSS was 231,620,608 bytes. This profile excludes
process startup and model loading, so it is a local comparison point rather than a
deployment SLO.

The path-light comparison evidence is retained with the external benchmark run outside
Git; no local filesystem path or raw question text is published here.

## Historical Protocol (superseded)

- Source: Kaggle Question Pairs Dataset v2, `kaggle://quora/question-pairs-dataset@2`
- Terms: Quora/Kaggle terms reviewed for this run; non-commercial use; raw data is not
  redistributed here.
- Source SHA-256: `6c821a0b8f6eec15eba08ffe24969c35dc414ad21372c845069d0e0d5ccd0f26`
- Split: question-component holdout, seed `42`, four roles at 70/10/10/10 percent.
- Train role SHA-256: `1d2c50841f6a32f865ad6b093215e97c5db3b13b063dc5423bda60be2b821453`
- Tuning role SHA-256: `328ca69a677ae01b1aee13d06e084d0f42c2b2ed641f10dc3d2ddcb632afc403`
- Calibration role SHA-256: `3a3a8fef4292349bfe60c3d0c3cadd6d2717baca5cf064dcc320ca193715f628`
- Final holdout SHA-256: `08a8e01733db5f42b2a4eb27104b2bf952da15a3b9d6b0b86eae2ab8c5c6e68d`
- Final holdout: 40,435 rows; positive rate 0.3711.
- Evaluation/report code: commit `2f463825f6a6952f0e3328586a1759d2b0dc12a3`.

The freezer was repeated from the same source bytes. All four role file hashes matched
exactly; only provenance fields such as the producing commit differed. Protocol and
protocol-bound report validation passed.

## Retrospective Data-Quality Note

The role set above is retained as historical evidence, but it is not promotion-grade under
the current strict benchmark policy: a direct cross-encoder dry run found one missing
`question2` value in the calibration role. The run failed closed before producing a
cross-encoder result; no quality or cost claim is made for that historical candidate.
Commits `a90b310` and `9087ded` now make the freezer, protocol builder, training lineage, and
report validator reject missing or blank question text. The corrected strict v2 comparison
above is the current evidence; the historical control and embedding numbers below should
not be read as evidence that the defective role is acceptable for a new promotion.

## Historical Pair Classification (superseded)

All rows below use the untouched final holdout. Thresholds were selected on the tuning role;
the final holdout was not used for model or threshold selection.

| candidate | threshold | ROC-AUC | PR-AUC | log loss | Brier | ECE | precision | recall | F1 | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + XGBoost control | 0.30 | 0.7888 | 0.6338 | 0.5091 | 0.1765 | 0.0052 | 0.5339 | 0.9138 | 0.6740 | 0.6719 |
| TF-IDF + sigmoid calibration | 0.30 | 0.7888 | 0.6338 | 0.5188 | 0.1779 | 0.0327 | 0.5516 | 0.8653 | 0.6737 | 0.6890 |
| TF-IDF + isotonic calibration | 0.30 | 0.7886 | 0.6270 | 0.5157 | 0.1766 | 0.0047 | 0.5319 | 0.9183 | 0.6736 | 0.6698 |
| Sentence-transformer embeddings + XGBoost | 0.40 | 0.8816 | 0.7871 | 0.4060 | 0.1352 | 0.0026 | 0.6708 | 0.8496 | 0.7497 | 0.7895 |

The embedding candidate is the strongest measured quality path in this run: its final
ROC-AUC is 0.8816 and F1 is 0.7497, versus 0.7888 and 0.6740 for the lexical control. The
model used was `sentence-transformers/all-MiniLM-L6-v2`; it was trained only on the train
role and selected with the tuning role. This is a measured candidate result, not a universal
quality guarantee for other domains or languages.

Calibration was not treated as automatically beneficial. On the calibration role, isotonic
changed Brier score from 0.1762 to 0.1758 and ECE from 0.0052 to approximately 0.0000. On
the final holdout it slightly improved ECE over the raw control, but worsened log loss and F1.
Sigmoid calibration worsened both Brier score and ECE in this run. The calibration methods
remain available as explicit artifacts; neither is silently presented as a global winner.

The cross-encoder candidate was not run on the historical role set above. No claim should
be inferred from that historical omission; the corrected strict v2 evidence is recorded at
the top of this page.

## Retrieval Cost Sample

The retrieval benchmark used a deterministic 1,000-query sample from the final holdout,
with 1,347 positive qrels and a 14,892-question catalog. It is a real-data performance
sample, not a full-catalog capacity guarantee.

- TF-IDF recall@1 / @5 / @10: `0.3066` / `0.5435` / `0.6549`
- MRR@10: `0.4406`; NDCG@10: `0.4824`
- Warm serial latency: p50 `6.82 ms`, p95 `7.98 ms`, p99 `9.86 ms`
- Throughput: `143.31` queries/second at concurrency 1
- Retriever initialization: `244.46 ms`
- Process peak RSS: `188,661,760` bytes
- Catalog SHA-256: `fd4472078cf7d0305edfdc6588bf3ced6b75a22fb9b07ae57e35f9063d8fab35`
- Qrels SHA-256: `19327344230772aecdef8d2370b0eb77a7c1b2bd56ed474776a532f81f34b247`

A first attempt over all 13,046 unique final-holdout queries exceeded the 180-second
cooperative deadline. That boundary is retained as a limitation rather than replaced with
an extrapolated capacity claim.

## Release Reading

- Use TF-IDF as the cheap, inspectable control.
- Treat the direct cross-encoder as the current quality candidate, with explicit latency,
  memory, artifact, and domain validation still required before promotion.
- Keep the embedding backend as a lower-cost intermediate candidate for latency and domain
  validation; it remains stronger than the historical lexical control.
- Keep threshold policy and calibration method explicit per deployment; do not assume the
  Quora calibration result transfers to customer data.
- Treat multilingual/domain slices, cross-encoder artifact packaging, and production load
  tests as the next release tasks.

See [REPORTING.md](REPORTING.md) for the commands and [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)
for the release gate.
