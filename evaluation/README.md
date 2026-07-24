# Evaluation source

- `evaluate_deepbee_official.py`: 13-image official DeepBee benchmark.
- `evaluate_fairhive10.py`: ten-image FAIRHive benchmark.
- `evaluate_inhouse6.py`: six-image in-house benchmark, including GeoComb's
  native eight-class and common seven-class evaluations.
- `bootstrap_detection_metrics.py`: image-cluster bootstrap for pooled
  precision, recall, F1, and mean image-level count error.

All spatial evaluators use one-to-one matching and the 0.5-pitch threshold
reported in the manuscript.

To reproduce the final benchmark intervals:

```bash
python evaluation/bootstrap_detection_metrics.py \
  --input reproducibility_data/final_benchmarks/bootstrap/final_benchmark_per_image_detection.csv \
  --output-method-ci method_ci.csv \
  --output-paired-ci paired_ci.csv
```

To reproduce the Table 2 intervals, use
`reproducibility_data/table2_component_analysis/table2_final_per_image_metrics.csv`
as the input. The defaults are 100,000 resamples and seed 20260721.
