CLEAN paper package drop-in replacement
======================================

1) Delete your existing paper_pkg/ (or move it to legacy/).
2) Copy the included paper_pkg/ folder into your repo root.

Then run (from repo root):
  python run_paper_pipeline.py --scenario paper_stress --preset balanced --epochs 3

Final run example:
  python run_paper_pipeline.py --scenario paper_stress --preset balanced --epochs 30 --max_sats 100 --seeds 0 1 2 3 4 5 6 7 8 9

Files created:
  results_paper/weights_tc_<scenario>.pth
  results_eval_<scenario>/exp_paper_runs.csv
  results_eval_<scenario>/exp_paper_summary.csv
  results_eval_<scenario>/paper_figures/*.png
