# Evaluation Harness (Offline)

## Goal
Track retrieval quality across:
- BM25 only (Meilisearch),
- ANN only (Qdrant),
- Hybrid RRF,
- Hybrid RRF + rerank.

Metrics: nDCG@10, Recall@20.

## Data
`evaluation/gold.csv` (edit freely):

```
query,robot_model,audience_level,category,doc_hint
"how to torque m3 screw on lidar bracket",S50,technician,sop,"Lidar Calibration v1.4"
"emergency stop reset after collision",generic,operator,safety,"Safety SOP 2025"
"wire map for ipc to motor driver v40",V40,technician,hardware,"Wiring Diagram v40"
```

## Procedure (suggested)
1. Run each retrieval variant and dump top-k ids + ranks per query to CSV.
2. Use `scikit-learn` or a small script to compute nDCG@10 and Recall@20.
3. Keep results in `evaluation/results/*.json` for trend tracking.

## Acceptance for demo
- RRF >= BM25 and ANN on both metrics.
- Rerank improves top-3 precision on procedural queries.
