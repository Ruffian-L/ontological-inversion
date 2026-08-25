# Memory-support calibration launch

The failure Jason saw was environmental, not experimental: `/usr/bin/python3` is
3.14 and had no NumPy. The machine also carries Python 3.12 and `uv`, so the run
now uses an ignored repository-local `.venv` built from the existing pinned
requirements. Sixty-nine packages were installed; no package versions or model
coordinates were changed.

The backend needed hardening before a 15,840-cell campaign. It previously loaded
Qwen on CPU even with a GB10 present, recalculated identical memory directions for
every probe, could repeat completed cells after interruption, and omitted the
attention mask despite Qwen using the same pad and EOS token. It now selects
CUDA/bfloat16 automatically, caches direction tensors by arm/source/seed, skips
cell IDs already present in the append-only output, and passes the tokenizer's
attention mask.

Two fresh one-cell smokes used the first locked calibration cell: matched arm,
gain 0, seed 104729, reconstruction probe. The correct source was “Ilya Venn
maintains the copper greenhouse in Oris Bay.” Both outputs instead began with an
unrelated request to inject four bytes at a hexadecimal address. At gain zero the
hook applies no memory direction, so failure to reconstruct is the expected
negative control. It is not evidence for or against readability at nonzero gain.

The full process launched as PID 126493 and emitted its first receipt. It writes
model progress to `results/memory_support_calibration_run.log` and responses to
`results/memory_support_calibration_responses.jsonl`. The run is resumable, so a
machine or process interruption need not invalidate completed cells.

The operational log initially exposed only opaque cell hashes. The added
`memory_support.watch` viewer follows the same receipt file and renders the source
memory, probe, arm, gain, seed, and complete output. It is read-only and does not
alter selection inputs or restart generation.

No judge has seen an output. Held-out remains locked. The next evidentiary boundary
is completion of all calibration receipts, followed by arm/gain-blinded semantic
judging and only then calibration-threshold freezing.
