"""Export and verify the frozen synthetic packet traces used in the study."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from iot_scheduler_verified import HORIZON_US, TEST_SEEDS, TRAIN_SEEDS, generate_trace


ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "traces"


def encoded_trace(seed: int, partition: str) -> bytes:
    payload = {
        "arrival_rate_packets_per_second": 300,
        "horizon_us": HORIZON_US,
        "packets": [asdict(packet) for packet in generate_trace(seed)],
        "partition": partition,
        "schema_version": 1,
        "seed": seed,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    TRACE_DIR.mkdir(exist_ok=True)
    manifest = {"schema_version": 1, "traces": []}
    for partition, seeds in (("train", TRAIN_SEEDS), ("test", TEST_SEEDS)):
        for seed in seeds:
            path = TRACE_DIR / f"{partition}_{seed}.json"
            blob = encoded_trace(seed, partition)
            if path.exists() and path.read_bytes() != blob:
                raise ValueError(f"Existing trace differs from generator: {path}")
            path.write_bytes(blob)
            manifest["traces"].append({
                "file": path.name,
                "packet_count": len(json.loads(blob)["packets"]),
                "partition": partition,
                "seed": seed,
                "sha256": hashlib.sha256(blob).hexdigest(),
            })
    manifest_path = TRACE_DIR / "trace_hashes.json"
    manifest_blob = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_blob:
        raise ValueError(f"Existing trace manifest differs: {manifest_path}")
    manifest_path.write_bytes(manifest_blob)
    print("Verified frozen traces:", ", ".join(
        f"{r['partition']} {r['seed']} ({r['packet_count']})" for r in manifest["traces"]
    ))


if __name__ == "__main__":
    main()
