# SIRCL Boundary Analysis

## Accuracy

- Baseline `simcon`: 53.60% (1319 samples)
- +SIRCL `simcon_sircl`: 56.56% (1319 samples)
- Absolute gain: +2.96%
- Relative gain: +5.52%

## Gain By Baseline Correctness

| Bucket | Count | Baseline Acc | +SIRCL Acc | Delta |
| --- | ---: | ---: | ---: | ---: |
| Baseline correct | 707 | 100.00% | 88.68% | -11.32% |
| Baseline wrong | 612 | 0.00% | 19.44% | +19.44% |
| All | 1319 | 53.60% | 56.56% | +2.96% |

## Transition Counts

| Transition | Count | Share |
| --- | ---: | ---: |
| Both correct | 627 | 47.54% |
| Recovered by +SIRCL | 119 | 9.02% |
| Both wrong | 493 | 37.38% |
| Regressed with +SIRCL | 80 | 6.07% |

## Geometry Highlights (Baseline Trajectories)

- Conditioned on baseline correctness, +SIRCL changes accuracy by +19.44% on baseline-wrong samples and -11.32% on baseline-correct samples.
- Baseline-wrong trajectories have radius_mean 11.373 vs 13.112 for baseline-correct ones.
- Baseline-wrong trajectories have token diversity 18.202 vs 20.714 for baseline-correct ones.
- Among baseline-wrong samples, recovered cases show radius_mean 11.767 vs 11.538 for still-wrong cases.
- Among baseline-wrong samples, recovered cases show token diversity 18.731 vs 18.430 for still-wrong cases.
