"""Auditable packet scheduler experiment. Python >=3.10; numpy required.

Run: python iot_scheduler_verified.py --self-test
     python iot_scheduler_verified.py --output results
     python iot_scheduler_verified.py --output results --quantum

The optional quantum run requires a separately installed compatible Qiskit SDK.
No published performance numbers are embedded in this program.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

HORIZON_US = 5_000_000
CAPACITY_BPS = 1_000_000
CLASSES = (0, 1, 2)
TERMINAL = ("TIMELY", "LATE", "DROPPED_EXPIRY", "DROPPED_SHED", "PENDING")
TRAIN_SEEDS = (101, 202)
TEST_SEEDS = (777, 888, 999)


@dataclass(frozen=True)
class Packet:
    id: int
    cls: int
    arrival_us: int
    size_bits: int
    deadline_us: int


@dataclass(frozen=True)
class Action:
    kind: str  # packet, class, or shed
    value: int = -1


@dataclass(frozen=True)
class Experience:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    next_actions: tuple[int, ...]
    done: bool


def generate_trace(seed: int, horizon_us: int = HORIZON_US,
                   arrival_rate: float = 300.0) -> tuple[Packet, ...]:
    rng = np.random.RandomState(seed)
    packets = []
    arrival_s = 0.0
    while True:
        arrival_s += rng.exponential(1.0 / arrival_rate)
        arrival_us = round(arrival_s * 1_000_000)
        if arrival_us >= horizon_us:
            break
        cls = int(rng.choice(CLASSES, p=[0.2, 0.5, 0.3]))
        size = int(rng.randint(2_000, 8_000))
        slack_s = rng.uniform(0.01, 0.04) if cls == 0 else rng.uniform(0.05, 0.15)
        packets.append(Packet(len(packets), cls, arrival_us, size,
                              arrival_us + round(slack_s * 1_000_000)))
    return tuple(packets)


def edf(buffer: tuple[Packet, ...], now_us: int) -> Action:
    return Action("packet", min(buffer, key=lambda p: (p.deadline_us, p.id)).id)


def llf(buffer: tuple[Packet, ...], now_us: int) -> Action:
    return Action("packet", min(buffer, key=lambda p: (
        p.deadline_us - now_us - p.size_bits * 1_000_000 / CAPACITY_BPS,
        p.deadline_us, p.id)).id)


def threshold_shedding(buffer: tuple[Packet, ...], now_us: int) -> Action:
    if len(buffer) > 15 and any(p.cls == 2 for p in buffer):
        return Action("shed")
    return edf(buffer, now_us)


def feasible_class_actions(buffer: tuple[Packet, ...]) -> tuple[int, ...]:
    actions = [c for c in CLASSES if any(p.cls == c for p in buffer)]
    if any(p.cls == 2 for p in buffer):
        actions.append(3)
    return tuple(actions)


def state_vector(buffer: tuple[Packet, ...], now_us: int) -> tuple[float, ...]:
    q = [sum(p.cls == c for p in buffer) for c in CLASSES]
    slack = [min(1.0, max(0.0, np.mean([
        (p.deadline_us - now_us) / 500_000 for p in buffer if p.cls == c
    ]))) if q[c] else 0.0 for c in CLASSES]
    return tuple([min(1.0, n / 50) for n in q] + slack + [CAPACITY_BPS / 2_000_000])


class RandomCoverage:
    """Behavior policy for training only; samples feasible class and shed actions."""

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def __call__(self, buffer: tuple[Packet, ...], now_us: int) -> Action:
        return Action("class", int(self.rng.choice(feasible_class_actions(buffer))))


class Simulator:
    # At an equal timestamp, completion precedes expiry; decisions occur last.
    PRIORITY = {"ARRIVAL": 0, "TX_COMPLETE": 1, "DEADLINE_EXPIRY": 2,
                "DECISION_COMPLETE": 3}

    def __init__(self, trace: tuple[Packet, ...], horizon_us: int = HORIZON_US):
        self.packets = {p.id: p for p in trace}
        if len(self.packets) != len(trace) or any(p.cls not in CLASSES for p in trace):
            raise ValueError("Packet IDs must be unique and classes valid")
        self.horizon_us = horizon_us
        self.heap = []
        self.seq = 0
        self.status = {p.id: "NEW" for p in trace}
        self.buffer: set[int] = set()
        self.active: int | None = None
        self.pending: tuple | None = None
        self.decision_generation = 0
        self.canceled_decisions = 0
        self.now = 0
        self.tx_intervals: list[tuple[int, int, int]] = []
        self.experiences: list[Experience] = []
        self.awaiting_next_decision: tuple | None = None
        self.decision_logs: list[dict] = []
        for p in trace:
            self._event(p.arrival_us, "ARRIVAL", p.id)
            self._event(p.deadline_us, "DEADLINE_EXPIRY", p.id)

    def _event(self, when: int, kind: str, pid: int = -1) -> None:
        self.seq += 1
        heapq.heappush(self.heap, (when, self.PRIORITY[kind], self.seq, kind, pid))

    def _snapshot(self) -> tuple[Packet, ...]:
        return tuple(self.packets[pid] for pid in sorted(self.buffer))

    def _begin(self, policy: Callable, delay_us: int) -> None:
        if self.pending is not None or self.active is not None or not self.buffer:
            return
        snapshot = self._snapshot()
        state = state_vector(snapshot, self.now)
        if self.awaiting_next_decision is not None:
            prior_state, prior_action, reward = self.awaiting_next_decision
            self.experiences.append(Experience(prior_state, prior_action, reward,
                state, feasible_class_actions(snapshot), False))
            self.awaiting_next_decision = None
        start_ns = time.perf_counter_ns()
        action = policy(snapshot, self.now)
        wall_ns = time.perf_counter_ns() - start_ns
        if action.kind not in ("packet", "class", "shed"):
            raise ValueError(f"Invalid action {action}")
        if action.kind == "class" and action.value not in (0, 1, 2, 3):
            raise ValueError(f"Invalid class action {action}")
        actual_delay_us = (policy.decision_delay_us(snapshot, self.now, delay_us)
                           if hasattr(policy, "decision_delay_us") else delay_us)
        if actual_delay_us < 0:
            raise ValueError("Policy supplied negative delay")
        self.decision_generation += 1
        self.pending = (state, action, self.now, wall_ns, actual_delay_us)
        self._event(self.now + actual_delay_us, "DECISION_COMPLETE",
                    self.decision_generation)

    def _finish_decision(self, policy: Callable, delay_us: int) -> None:
        state, requested, started, wall_ns, actual_delay_us = self.pending
        self.pending = None
        shed = 0
        fallback = False
        chosen: int | None = None
        action_index = requested.value if requested.kind == "class" else -1
        if requested.kind == "shed" or action_index == 3:
            be = [pid for pid in self.buffer if self.packets[pid].cls == 2]
            if be:
                victim = min(be, key=lambda pid: (self.packets[pid].arrival_us, pid))
                self.buffer.remove(victim)
                self.status[victim] = "DROPPED_SHED"
                shed = 1
        elif requested.kind == "packet":
            if requested.value in self.buffer:
                chosen = requested.value
            else:
                fallback = True
        elif action_index in CLASSES:
            candidates = [pid for pid in self.buffer if self.packets[pid].cls == action_index]
            if candidates:
                chosen = min(candidates, key=lambda pid: (self.packets[pid].deadline_us, pid))
            else:
                fallback = True
        if chosen is None and self.buffer:
            chosen = edf(self._snapshot(), self.now).value
        self.decision_logs.append({"start_us": started, "complete_us": self.now,
                                   "wall_ns": wall_ns, "modeled_delay_us": actual_delay_us,
                                   "requested_kind": requested.kind,
                                   "requested_value": requested.value,
                                   "selected_packet": chosen if chosen is not None else "",
                                   "shed": shed, "fallback": fallback})
        if chosen is None:
            if action_index >= 0:
                self.awaiting_next_decision = (state, action_index, -1.0)
            return
        self.buffer.remove(chosen)
        self.status[chosen] = "IN_SERVICE"
        self.active = chosen
        duration = (self.packets[chosen].size_bits * 1_000_000 + CAPACITY_BPS - 1) // CAPACITY_BPS
        finish = self.now + duration
        self.tx_intervals.append((self.now, finish, chosen))
        self._event(finish, "TX_COMPLETE", chosen)
        # The reward is recorded on TX_COMPLETE, after the outcome is known.
        self.outstanding = (state, action_index, shed)

    def run(self, policy: Callable, delay_us: int = 0) -> dict:
        if delay_us < 0:
            raise ValueError("Decision delay must be nonnegative")
        while self.heap:
            when, _, _, kind, pid = heapq.heappop(self.heap)
            if when > self.horizon_us:
                break
            self.now = when
            if kind == "ARRIVAL":
                self.status[pid] = "QUEUED"
                self.buffer.add(pid)
                if (self.pending is not None and self.pending[4] > 0
                        and hasattr(policy, "preempt_for_arrival")
                        and policy.preempt_for_arrival(self.packets[pid])):
                    # The scheduled completion is invalidated by its generation.
                    self.pending = None
                    self.canceled_decisions += 1
            elif kind == "DEADLINE_EXPIRY":
                if self.status[pid] == "QUEUED":
                    self.buffer.remove(pid)
                    self.status[pid] = "DROPPED_EXPIRY"
            elif kind == "DECISION_COMPLETE":
                if self.pending is not None and pid == self.decision_generation:
                    self._finish_decision(policy, delay_us)
            elif kind == "TX_COMPLETE":
                if pid != self.active:
                    raise AssertionError("Transmission completion out of order")
                outcome = "TIMELY" if when <= self.packets[pid].deadline_us else "LATE"
                self.status[pid] = outcome
                self.active = None
                state, act, shed = self.outstanding
                if act >= 0:
                    # Reward bounded to [-1,1] for an expectation-value Q head.
                    reward = (1.0 if outcome == "TIMELY" else -0.5) - 0.5 * shed
                    self.awaiting_next_decision = (state, act, reward)
            self._begin(policy, delay_us)
        if self.awaiting_next_decision is not None:
            state, act, reward = self.awaiting_next_decision
            snapshot = self._snapshot()
            self.experiences.append(Experience(state, act, reward,
                state_vector(snapshot, self.horizon_us), (), True))
            self.awaiting_next_decision = None
        for pid in self.buffer:
            self.status[pid] = "PENDING"
        if self.active is not None:
            self.status[self.active] = "PENDING"
        counts = {key: sum(s == key for s in self.status.values()) for key in TERMINAL}
        assert set(self.status.values()) <= set(TERMINAL), self.status
        assert sum(counts.values()) == len(self.packets)
        intervals = sorted(self.tx_intervals)
        assert all(b[1] <= a[0] for b, a in zip(intervals, intervals[1:])), "Link overlap"
        busy_us = sum(max(0, min(end, self.horizon_us) - start)
                      for start, end, _ in intervals if start < self.horizon_us)
        assert busy_us <= self.horizon_us, "Link exceeds horizon capacity"
        assert counts["TIMELY"] + counts["LATE"] <= len(intervals)
        return {"total": len(self.packets), **counts, "link_busy_us": busy_us,
                "decisions": len(self.decision_logs), "delay_us": delay_us,
                "canceled_decisions": self.canceled_decisions,
                "fallbacks": sum(bool(x["fallback"]) for x in self.decision_logs)}


def action_angles(a: int) -> tuple[float, float]:
    return ((0, 0), (0, np.pi / 2), (np.pi / 2, 0),
            (np.pi / 2, np.pi / 2))[a]


class QuantumScorer:
    """Optional Qiskit V2 state-action scorer: nine input angles, 18 weights."""

    def __init__(self, theta: np.ndarray | None = None):
        from qiskit.circuit import QuantumCircuit, ParameterVector
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp
        self.x = ParameterVector("x", 9)
        self.w = ParameterVector("w", 18)
        self.qc = QuantumCircuit(9)
        for i in range(9):
            self.qc.rx(self.x[i], i)
        for layer in range(2):
            for i in range(9):
                self.qc.ry(self.w[layer * 9 + i], i)
            for i in range(8):
                self.qc.cx(i, i + 1)
        self.observable = SparsePauliOp("Z" + "I" * 8)
        self.estimator = StatevectorEstimator()
        self.theta = np.array(theta if theta is not None else np.zeros(18), dtype=float)

    def score(self, state: tuple[float, ...], action: int,
              theta: np.ndarray | None = None) -> float:
        x = np.r_[np.asarray(state) * np.pi, action_angles(action)]
        weights = self.theta if theta is None else theta
        bindings = {**dict(zip(self.x, x)), **dict(zip(self.w, weights))}
        circuit = self.qc.assign_parameters(bindings)
        return float(np.asarray(self.estimator.run([(circuit, self.observable)])
                                .result()[0].data.evs).item())

    def __call__(self, buffer: tuple[Packet, ...], now_us: int) -> Action:
        state = state_vector(buffer, now_us)
        actions = feasible_class_actions(buffer)
        return Action("class", max(actions, key=lambda a: self.score(state, a)))


class MatchedMLP:
    """Two tanh units with fixed average output: 2 x 9 = 18 trainable weights."""

    def __init__(self, theta: np.ndarray | None = None):
        self.theta = np.array(theta if theta is not None else np.zeros(18), dtype=float)

    def score(self, state: tuple[float, ...], action: int,
              theta: np.ndarray | None = None) -> float:
        x = np.r_[np.asarray(state) * np.pi, action_angles(action)]
        weights = self.theta if theta is None else theta
        return float(np.mean(np.tanh(np.asarray(weights).reshape(2, 9) @ x)))

    def __call__(self, buffer: tuple[Packet, ...], now_us: int) -> Action:
        state = state_vector(buffer, now_us)
        actions = feasible_class_actions(buffer)
        return Action("class", max(actions, key=lambda a: self.score(state, a)))


class UrgentClassGuard:
    """Safety architecture: class-0 packet bypasses the parametric scorer."""

    def __init__(self, scorer):
        self.scorer = scorer

    def decision_delay_us(self, buffer: tuple[Packet, ...], now_us: int,
                          nominal_delay_us: int) -> int:
        return 0 if any(p.cls == 0 for p in buffer) else nominal_delay_us

    def preempt_for_arrival(self, packet: Packet) -> bool:
        return packet.cls == 0

    def __call__(self, buffer: tuple[Packet, ...], now_us: int) -> Action:
        urgent = [p for p in buffer if p.cls == 0]
        if urgent:
            return edf(tuple(urgent), now_us)
        return self.scorer(buffer, now_us)


def train_scorer(model, experiences: list[Experience], seed: int = 42,
                 epochs: int = 5, batch_size: int = 32) -> list:
    if not experiences:
        raise ValueError("No real transition samples collected")
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-0.5, 0.5, 18)
    subset = [experiences[i] for i in rng.choice(len(experiences),
              size=min(batch_size, len(experiences)), replace=False)]
    history = []
    for epoch in range(epochs):
        # Freeze bootstrap targets for the entire epoch, including both SPSA probes.
        target_theta = theta.copy()
        targets = [e.reward + (0 if e.done or not e.next_actions else
                   0.8 * max(model.score(e.next_state, a, target_theta)
                             for a in e.next_actions)) for e in subset]
        def loss(candidate):
            return float(np.mean([(model.score(e.state, e.action, candidate) - y) ** 2
                                  for e, y in zip(subset, targets)]))
        before = loss(theta)
        delta = rng.choice([-1.0, 1.0], size=18)
        c, a = 0.05, 0.1 / (epoch + 1) ** 0.6
        grad = (loss(theta + c * delta) - loss(theta - c * delta)) / (2 * c) * delta
        theta -= a * grad
        history.append({"epoch": epoch + 1, "frozen_target_loss_before": before,
                        "frozen_target_loss_after": loss(theta)})
    model.theta = theta.copy()
    return history


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict]) -> None:
    keys = ("TIMELY", "LATE", "DROPPED_EXPIRY", "DROPPED_SHED", "PENDING")
    names = list(dict.fromkeys(row["policy"] for row in rows))
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Packet outcomes on held-out traces; decision delay is modeled.}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Policy & Timely & Late & Expiry & Shed & Pending & Total \\",
             r"\midrule"]
    for name in names:
        group = [r for r in rows if r["policy"] == name]
        values = [sum(r[k] for r in group) for k in keys]
        total = sum(r["total"] for r in group)
        assert sum(values) == total
        safe_name = name.replace("&", r"\&").replace("%", r"\%")
        lines.append(safe_name + " & " + " & ".join(map(str, (*values, total))) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def self_test() -> None:
    # Ten packets, overlapping deadlines, a deliberately busy single link.
    trace = tuple(Packet(i, i % 3, i * 400, 2_000 + i * 100,
                         i * 400 + 3_000) for i in range(10))
    for policy in (edf, llf, threshold_shedding, RandomCoverage(7)):
        sim = Simulator(trace, horizon_us=20_000)
        row = sim.run(policy, delay_us=1_000)
        assert sum(row[k] for k in TERMINAL) == 10
        assert row["link_busy_us"] <= 20_000
    # A delayed decision can expire; expiry must precede any stale selection.
    short = (Packet(0, 0, 0, 2_000, 500),)
    sim = Simulator(short, horizon_us=5_000)
    row = sim.run(edf, delay_us=1_000)
    assert row["DROPPED_EXPIRY"] == 1 and row["TIMELY"] == 0
    # LLF must return the exact packet, even if it is not the class EDF head.
    packets = (Packet(0, 0, 0, 2_000, 10_000), Packet(1, 0, 0, 8_000, 11_000))
    assert llf(packets, 0).value == 1
    assert len({action_angles(a) for a in range(4)}) == 4
    interrupted = (Packet(0, 2, 0, 2_000, 100_000),
                   Packet(1, 0, 1_000, 2_000, 12_000))
    guarded = Simulator(interrupted, horizon_us=100_000)
    result = guarded.run(UrgentClassGuard(lambda b, t: Action("class", 2)),
                         delay_us=15_000)
    assert result["canceled_decisions"] == 1
    assert guarded.status[1] == "TIMELY"
    slow = Simulator(generate_trace(777))
    slow_result = slow.run(lambda b, t: Action("class", feasible_class_actions(b)[0]),
                           delay_us=15_000)
    assert slow_result["TIMELY"] + slow_result["LATE"] <= 333
    print("PASS: conservation, link exclusion, delayed expiry, packet LLF, encoding")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quantum", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    policies = {"EDF": edf, "LLF": llf, "Threshold Best-Effort Shedding": threshold_shedding}
    training = []
    for seed in TRAIN_SEEDS:
        sim = Simulator(generate_trace(seed))
        # Match the deployed safety architecture and nonurgent delay. The
        # random scorer explores only feasible nonurgent actions; urgent
        # packets bypass it through the same guard used at evaluation.
        sim.run(UrgentClassGuard(RandomCoverage(seed)), delay_us=15_000)
        training.extend(sim.experiences)
    mlp = MatchedMLP()
    mlp_history = train_scorer(mlp, training)
    np.save(args.output / "frozen_mlp_weights.npy", mlp.theta)
    write_csv(args.output / "mlp_training_loss.csv", mlp_history)
    policies["Matched MLP (0 ms modeled delay)"] = (mlp, 0)
    policies["Matched MLP (15 ms modeled delay)"] = (mlp, 15_000)
    policies["Guarded MLP (15 ms nonurgent delay)"] = (UrgentClassGuard(mlp), 15_000)
    if args.quantum:
        model = QuantumScorer()
        history = train_scorer(model, training)
        np.save(args.output / "frozen_theta.npy", model.theta)
        write_csv(args.output / "training_loss.csv", history)
        policies["Trained VQC (0 ms modeled delay)"] = (model, 0)
        policies["Trained VQC (15 ms modeled delay)"] = (model, 15_000)
        policies["Guarded VQC (15 ms nonurgent delay)"] = (UrgentClassGuard(model), 15_000)
    rows = []
    class_rows = []
    for seed in TEST_SEEDS:
        trace = generate_trace(seed)
        for name, entry in policies.items():
            policy, delay = entry if isinstance(entry, tuple) else (entry, 0)
            sim = Simulator(trace)
            row = {"seed": seed, "policy": name, **sim.run(policy, delay)}
            wall_ms = np.array([d["wall_ns"] / 1_000_000 for d in sim.decision_logs])
            row["mean_local_scorer_ms"] = float(np.mean(wall_ms)) if len(wall_ms) else 0.0
            row["p95_local_scorer_ms"] = float(np.percentile(wall_ms, 95)) if len(wall_ms) else 0.0
            rows.append(row)
            for cls in CLASSES:
                outcomes = {key: sum(sim.status[pid] == key for pid, p in sim.packets.items()
                                     if p.cls == cls) for key in TERMINAL}
                class_total = sum(p.cls == cls for p in sim.packets.values())
                assert sum(outcomes.values()) == class_total
                class_rows.append({"seed": seed, "policy": name, "class": cls,
                                   "total": class_total, **outcomes})
            print(row)
            if args.quantum and (name.startswith("Trained VQC") or name.startswith("Guarded VQC")):
                suffix = ("guarded" if name.startswith("Guarded") else
                          "15ms" if delay else "0ms")
                write_csv(args.output / f"vqc_decisions_{seed}_{suffix}.csv",
                          sim.decision_logs)
    write_csv(args.output / "per_seed_results.csv", rows)
    write_csv(args.output / "per_class_results.csv", class_rows)
    write_latex(args.output / "results_table.tex", rows)
    urgent = [p for seed in TEST_SEEDS for p in generate_trace(seed) if p.cls == 0]
    isolated_margins = sorted(p.deadline_us - p.arrival_us -
        math.ceil(p.size_bits * 1_000_000 / CAPACITY_BPS) for p in urgent)
    target_count = math.ceil(0.95 * len(urgent))
    (args.output / "sla_feasibility.json").write_text(json.dumps({
        "interpretation": "Optimistic no-contention per-packet bound; not an achievable schedule",
        "urgent_total": len(urgent), "target_timely_95pct": target_count,
        "isolated_feasible_with_15ms_each": sum(m >= 15_000 for m in isolated_margins),
        "largest_delay_for_95pct_isolated_feasibility_us":
            isolated_margins[len(urgent) - target_count]}, indent=2) + "\n")
    (args.output / "run_manifest.json").write_text(json.dumps({
        "train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
        "horizon_us": HORIZON_US, "capacity_bps": CAPACITY_BPS,
        "quantum_executed": args.quantum,
        "qiskit_version": (__import__("qiskit").__version__ if args.quantum else None),
        "numpy_version": np.__version__,
        "python_version": __import__("platform").python_version(),
        "platform": __import__("platform").platform(),
        "quantum_decision_delay_us": 15_000 if args.quantum else None,
        "training_behavior": "class-0 guard plus random feasible nonurgent actions",
        "training_nonurgent_delay_us": 15_000,
        "model_trainable_parameters": 18,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_experiences": len(training)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
