"""Independent archive consistency and optional deterministic rerun gate.

python verify_iot_experiment.py --results experiment_results
python verify_iot_experiment.py --results experiment_results --rerun
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from iot_scheduler_verified import (CLASSES, TERMINAL, QuantumScorer,
                                    generate_trace, write_latex)

COUNT_FIELDS = (*TERMINAL, "total", "link_busy_us", "decisions",
                "delay_us", "fallbacks", "canceled_decisions")


def records(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def keyed(rows, fields):
    result = {tuple(r[k] for k in fields): r for r in rows}
    assert len(result) == len(rows), f"Duplicate keys: {fields}"
    return result


def validate(directory, minimum_sla):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    source = Path(__file__).with_name("iot_scheduler_verified.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    assert set(manifest["train_seeds"]).isdisjoint(manifest["test_seeds"])
    assert manifest["training_nonurgent_delay_us"] == 15_000
    assert manifest["model_trainable_parameters"] == 18
    assert manifest["training_experiences"] > 0
    rows = records(directory / "per_seed_results.csv")
    classes = keyed(records(directory / "per_class_results.csv"),
                    ("seed", "policy", "class"))
    assert len(rows) == len(set((r["seed"], r["policy"]) for r in rows))
    for r in rows:
        seed = int(r["seed"])
        trace = generate_trace(seed)
        assert seed in manifest["test_seeds"]
        assert int(r["total"]) == len(trace)
        assert sum(int(r[k]) for k in TERMINAL) == len(trace)
        assert 0 <= int(r["link_busy_us"]) <= manifest["horizon_us"]
        assert int(r["TIMELY"]) + int(r["LATE"]) <= int(r["decisions"])
        for c in CLASSES:
            cr = classes[(r["seed"], r["policy"], str(c))]
            assert int(cr["total"]) == sum(p.cls == c for p in trace)
            assert sum(int(cr[k]) for k in TERMINAL) == int(cr["total"])
        for k in TERMINAL:
            assert int(r[k]) == sum(int(classes[(r["seed"], r["policy"], str(c))][k])
                                   for c in CLASSES)
        if ("15 ms modeled delay" in r["policy"]
                and not r["policy"].startswith("Guarded")):
            # Each decision is serial and spends at least 15 ms before service.
            assert int(r["decisions"]) <= math.ceil(manifest["horizon_us"] / 15_000)
        if r["policy"].startswith("Guarded"):
            urgent = classes[(r["seed"], r["policy"], "0")]
            rate = int(urgent["TIMELY"]) / int(urgent["total"])
            print(f"{r['policy']} seed {seed}: class-0 SLA={rate:.4%}")
            assert rate >= minimum_sla, f"Observed SLA failure: {r['policy']} seed {seed}"

    assert len(classes) == len(rows) * len(CLASSES)
    with tempfile.TemporaryDirectory() as temp:
        rendered = Path(temp) / "results_table.tex"
        numerical = [{k: (int(v) if k in (*TERMINAL, "total") else v)
                      for k, v in r.items()} for r in rows]
        write_latex(rendered, numerical)
        assert rendered.read_bytes() == (directory / "results_table.tex").read_bytes()

    urgent = [p for s in manifest["test_seeds"] for p in generate_trace(s) if p.cls == 0]
    possible = sum(p.deadline_us - p.arrival_us >= 15_000 + p.size_bits
                   for p in urgent)  # 1 Mbps => one microsecond per bit.
    feasibility = json.loads((directory / "sla_feasibility.json").read_text())
    assert feasibility["urgent_total"] == len(urgent)
    assert feasibility["isolated_feasible_with_15ms_each"] == possible
    assert feasibility["target_timely_95pct"] == math.ceil(.95 * len(urgent))

    mlp = np.load(directory / "frozen_mlp_weights.npy")
    assert mlp.shape == (18,) and np.all(np.isfinite(mlp))
    if manifest["quantum_executed"]:
        theta = np.load(directory / "frozen_theta.npy")
        assert theta.shape == (18,) and np.all(np.isfinite(theta))
        scorer = QuantumScorer(theta)
        sample = (.2, .3, .1, .4, .5, .3, .5)
        scores = np.array([scorer.score(sample, a) for a in range(4)])
        assert np.all(np.isfinite(scores)) and np.ptp(scores) > 1e-8
    print(f"PASS: {len(rows)} policy/seed rows, {len(classes)} class rows, "
          f"table, conservation, capacity, seeds, feasibility, model weights")
    return rows, manifest


def rerun(directory, reference_rows, manifest):
    with tempfile.TemporaryDirectory() as temp:
        cmd = [sys.executable, str(Path(__file__).with_name("iot_scheduler_verified.py")),
               "--output", temp]
        if manifest["quantum_executed"]:
            cmd.append("--quantum")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        new_rows, _ = validate(Path(temp), minimum_sla=0.95)
        old = keyed(reference_rows, ("seed", "policy"))
        new = keyed(new_rows, ("seed", "policy"))
        assert old.keys() == new.keys()
        for key in old:
            for field in COUNT_FIELDS:
                assert old[key][field] == new[key][field], (key, field,
                                                           old[key][field], new[key][field])
        print("PASS: fresh rerun matches every archived packet and decision count")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--sla-min", type=float, default=.95)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    rows, manifest = validate(args.results, args.sla_min)
    if args.rerun:
        rerun(args.results, rows, manifest)


if __name__ == "__main__":
    main()
