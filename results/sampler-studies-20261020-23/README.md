# UQ proxy and MGKDB sampler study archive

Archived 2026-10-23 from the original local Codex study artifacts. This directory is the durable repository copy; no experiment was rerun for archiving.

## Reports

- [Ionut/UQ proxy report](proxy-study-20261020/index.html): 440/440 trajectories complete; historical sglib, fixed-case development evidence, paired inference pending.
- [MGKDB model development](mgkdb-stage2-20261021/report/index.html): nested predictor search and diagnostics.
- [MGKDB sampler learning curves](sampler-learning-20261021/index.html): 160/160 trajectories complete.
- [Paired sampler inference](sampler-learning-20261021/paired_analysis.html): one of 42 contrasts cleared both registered criteria; no overall winner.
- [Percentage extension snapshot](percentage-learning-20261022/index.html): INCOMPLETE, 22/160 trajectories complete at inspection; saved status.json says running but recorded parent process and Python workers were absent. Do not treat it as a live report or compare unequal completion counts as final results.

## Provenance and reuse

Protocols, runner/analysis sources, logs, PID records, prepared arrays, sampled orders, predictions, scores, figures and original manifests are preserved. PID files are historical, not proof of running processes. Original scripts and manifests intentionally retain original environment paths and hashes; running them on another machine requires inspecting/configuring their data and import paths. They are archived execution sources, not newly installed package commands. Raw MGKDB source pickles remain in the classifier repository; available prepared arrays are included here.

`archive-manifest.json` records every source artifact's byte count and SHA-256, the archive path and stored SHA-256. Files are byte-identical except the oversized proxy `results.json`, stored as `proxy-study-20261020/results.json.gz` (lossless gzip). Python readers can use `json.load(gzip.open(path, 'rt', encoding='utf-8'))`; decompression reproduces the source hash. Reports use local figures. Snapshot data and model files are research artifacts, not independent confirmation of transfer to NSTX.

Do not rerun archived scripts blindly: some write beside themselves. Copy to a new run location and register any new protocol before execution. TabPFN regression and removal ablations are not present because they were not run.
