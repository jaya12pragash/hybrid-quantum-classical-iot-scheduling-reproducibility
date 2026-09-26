# Hybrid Quantum–Classical IoT Scheduling: Reproducibility

Companion materials for **Hybrid Quantum--Classical Policy Scoring for Deadline-Constrained IoT Traffic Scheduling in 6G-Oriented Edge Networks** by Jayapragash Dakshnamurthy. This repository contains the synthetic packet traces, archived outputs, frozen model weights, and code for auditing the experiment. It does not claim physical quantum hardware or a deployed 6G network.

## Contents

- `data/iot_scheduler_verified.py`: integer-microsecond event engine and scorers.
- `data/verify_iot_experiment.py`: validates archived results; `--rerun` repeats the Qiskit-dependent experiment.
- `data/export_traces.py` and `data/traces/`: synthetic packet records and SHA-256 manifest.
- `data/frozen_theta.npy`, `data/frozen_mlp_weights.npy`: frozen trained parameters.
- `data/per_seed_results.csv`, `data/per_class_results.csv`: archived outcomes.
- `data/vqc_decisions_*.csv`: archived decisions.
- `data/run_manifest.json`, `data/requirements.txt`: settings and pinned Python packages.
- `make_assets.py`, `make_structural_figures.py`: manuscript figures and tables (expect the companion manuscript directory structure).

## Environment and verification

Use Python 3.12 and install `data/requirements.txt` in an isolated environment. Then run:

```bash
python data/iot_scheduler_verified.py --self-test
python data/verify_iot_experiment.py --results data --rerun
```

The verifier requires Qiskit even without `--rerun` because it checks the frozen scorer. Reproducing the archived training also requires the pinned environment. The archived data record three held-out seeds (777, 888, 999) and one training initialization per scorer. The 15 ms decision delay is modeled event time, not measured quantum hardware latency. A shared classical urgent guard accounts for the urgent-class outcome; the small nonurgent difference is exploratory.

No license is granted by this package. Contact the author before reuse beyond inspection and verification.
