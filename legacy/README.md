# paper_pkg (new paper experiment suite)

This folder contains ONLY the new planned experiments for your paper.
Legacy scripts (Exp1/Exp2 etc.) are not required.

## Entry points

### Train TC (Transformer+Consistency)
Run one of the presets (stable/balanced/aggressive):

```bash
python -m paper_pkg.train --scenario paper_stress --results_dir results_paper --preset balanced
```

Weights are saved to:
- results_paper/weights_tc_<scenario>.pth
and a manifest:
- results_paper/train_manifest_<scenario>.json

### Evaluate (baselines + optional TC)
With TC:
```bash
python -m paper_pkg.eval --scenario paper_stress --results_dir results_eval --seeds 8 9 10 --weights results_paper/weights_tc_paper_stress.pth
```

Baselines only:
```bash
python -m paper_pkg.eval --scenario paper_stress --results_dir results_eval --seeds 8 9 10
```

Outputs:
- results_eval/exp_paper_runs.csv
- results_eval/exp_paper_summary.csv

### Make figures
```bash
python -m paper_pkg.figures --summary results_eval/exp_paper_summary.csv --outdir results_eval/paper_figures --scenario paper_stress
```

## What is included
- env_core.py        : env builder + runtime/execution simulator + DAG baseline
- baselines.py       : reactive / lookahead / shooting-search baselines (with guardrails)
- tc_train_eval.py   : TC train/eval core (import-safe, no main)
- train.py / eval.py : paper runners
- figures.py         : paper-ready PNG export

