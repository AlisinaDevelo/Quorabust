# Real Benchmark Result

This page records one permitted Quora Question Pairs evaluation run. It is evidence for
review, not a claim of state-of-the-art quality. The raw dataset and frozen role CSVs stay
outside Git.

## Protocol

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

## Pair Classification

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

The cross-encoder candidate was not run on the full frozen protocol in this release. No
cross-encoder quality claim is made.

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
- Use the embedding backend as the quality candidate for further latency, artifact, and
  domain validation.
- Keep threshold policy and calibration method explicit per deployment; do not assume the
  Quora calibration result transfers to customer data.
- Treat the cross-encoder and multilingual/domain slices as follow-up research until they
  have permitted, frozen evaluation data.

See [REPORTING.md](REPORTING.md) for the commands and [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)
for the release gate.
