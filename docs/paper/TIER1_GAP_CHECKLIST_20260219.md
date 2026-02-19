# Tier-1 gap checklist (SSOT: nlp8/results_paper)

Updated: 2026-02-19 21:05 KST

## Scope
This checklist is derived from `ssh nlp8:/data_x/aa007878/galileo/results_paper/tier1_*`.
A run is marked **OK** only if `paper_exports/` has all required files:
- survival_curve.csv
- turn_of_failure.csv
- flip_samples.csv
- metadata.json
- runner_metadata.json

## Paper-ready (OK)
- tier1_llama3_3b_seed1_20260212_030426
- tier1_llama3_3b_seed2_20260212_042339
- tier1_phi3mini_seed1_20260217_011737
- tier1_phi3mini_seed2_20260217_033953
- tier1_mistralnemo_seed1_20260217_173907
- tier1_mistralnemo_seed2_20260217_180951
- tier1_zephyr7b_seed1_20260218_0945
- tier1_zephyr7b_seed2_20260218_141231
- tier1_qwen2p5_14b_seed1_20260219_032551
- tier1_qwen2p5_14b_seed2_20260219_053824
- tier1_deepseek7b_seed1_20260219_112728
- tier1_deepseek7b_seed2_20260219_112728
- tier1_phi35mini_seed1_20260219_143555
- tier1_phi35mini_seed2_20260219_143555

## Incomplete / non-citable (MISS)
- tier1_falcon7b_seed1_20260217_145044
- tier1_gemma2_2b_seed1_20260217_141927
- tier1_gemma2_2b_seed1_len4096_20260217_144011
- tier1_pythia2p8b_seed1_20260217_155743
- tier1_pythia2p8b_seed1_len2048_20260217_162421
- tier1_pythia2p8b_seed1_len2048_20260218_0424
- tier1_zephyr7b_seed1_20260217_150053
- tier1_zephyr7b_seed2_20260218_1034

## Decision rule (for 2026-02-28 closure)
1. Do **not** spend additional compute on known incompatible lines (Gemma2/Falcon on current nlp8 stack) unless env/backend changes are explicitly approved.
2. If we need extra confidence, prefer:
   - one clean additional family that is known to run on nlp8, or
   - seed extensions for already stable families only when confidence intervals are story-critical.
3. Keep `results_paper/` parity green after every addition.

## Immediate next launch policy
- Host: nlp8
- GPUs: 0–6, but only idle/not used by others at launch time.
- Prefer max 3 concurrent heavy jobs unless headroom verified.
