# Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner

- Year: 2025
- Venue: arXiv
- Authors: Guan-Ting Lin; Shih-Yun Shan Kuan; Jiatong Shi; Kai-Wei Chang; Siddhant Arora; Shinji Watanabe; Hung-yi Lee
- URL: https://arxiv.org/abs/2510.07838
- BibTeX key (if we add it): fullDuplexBenchV2_2025_lin
- Tags: multi-turn, evaluation, dialogue, full-duplex, speech, automated-examiner, drift

## One-sentence takeaway

FDB-v2 is a streaming, full-duplex speech-agent evaluation framework where an automated examiner drives staged multi-turn goals (fast vs. slow pacing) and an LLM judge scores turn-taking fluency, multi-turn instruction following, and task-specific competence, revealing degradation over time and sensitivity to pacing.

## What problem does it solve?

- Full-duplex (simultaneous speak+listen) speech agents are evaluated mostly in single-turn or scripted overlap cases; multi-turn, goal-driven interaction with interruptions/corrections/entity carry-over is under-evaluated.
- Human evaluation is expensive and hard to reproduce; existing automatic proxies (timing stats, classifiers) miss semantic/task success across turns.

## What is the core method / protocol?

- Introduces Full-Duplex-Bench-v2 (FDB-v2): an automated, streaming-native evaluation loop with three components:
  - Examiner: a spoken language model that roleplays scenarios and enforces stepwise semantic goals (staged subgoals).
  - Orchestrator: connects Examiner and Evaluatee over a standardized streaming protocol (full-duplex audio).
  - Evaluatee: the system under test (commercial API or open model).
- Two examiner pacing regimes:
  - Fast: more initiative; can interrupt/barge-in; advances stages quickly.
  - Slow: passive; responds after end-of-turn/long pause; no barge-in.
- Four task families (scenario sets):
  - Daily (routine task completion)
  - Correction (self-repairs / slot revisions across turns)
  - Entity Tracking (coreference / reference shifts)
  - Safety (policy-aligned hazardous requests, refusal/redirection)
- Scoring pipeline:
  - Record dual-channel audio, transcribe with ASR.
  - Use an LLM judge conditioned on the examiner system prompt + stage goals to score turn-taking/instruction-following per event, plus a global task-specific score.

## What are the key metrics?

- Turn-Taking Fluency (1-5), per turn-taking event (overlap/handoff quality).
- Multi-Turn Instruction Following (1-5), per event (adherence to staged goals).
- Task-specific global score (1-5), depending on family:
  - Entity tracking consistency
  - Correction handling quality
  - Safety boundary + consistent refusal/redirection under overlap/pressure
- They also present trajectories over time (e.g., aggregated into 15s bins) to quantify temporal degradation/drift.

## What are the main results?

- Across tested full-duplex spoken-LLM systems, performance degrades over the course of a session:
  - Turn-taking tends to drift down more steadily.
  - Instruction-following is more volatile (drops early, occasional partial recovery).
- Examiner pacing matters:
  - Slow pacing often stabilizes and can improve instruction-following (especially for entity tracking/correction), while fast pacing increases volatility and reduces recovery.
- Task difficulty: correction and entity tracking are highlighted as particularly challenging for open systems; safety appears more robust but still benefits from pacing.

## How is this similar to GALILEO?

- Both emphasize *multi-turn* interaction dynamics rather than single-turn accuracy.
- Both explicitly track *temporal degradation* / instability across turns and compare trajectories.
- Both benefit from having a controlled protocol (scripted roles/goals) that is reproducible.

## How is this different from GALILEO?

- Domain/setting: FDB-v2 is for *streaming speech* and full-duplex overlap phenomena; GALILEO is text-based multi-turn pressure on tasks with ground-truth answers.
- Evaluator: FDB-v2 relies heavily on an automated examiner + LLM-as-judge scoring (subject to prompt/judge calibration drift); GALILEO uses ground-truth tasks with explicit correctness tracking and a neutral re-asking control to isolate pressure vs. drift.
- Target failure modes: FDB-v2 focuses on turn-taking fluency, correction adoption, entity tracking, and safety under overlap; GALILEO focuses on correctness survival/TOF and recovery under persona pressure.

## Where GALILEO is stronger / cleaner (if true)

- Clear ground-truth correctness makes survival/TOF/recovery metrics more objective and easier to audit than 1-5 LLM-judge scores.
- Neutral re-asking control is a clean baseline to disentangle drift from adversarial pressure (FDB-v2 varies pacing but does not center an explicit drift-vs-pressure decomposition).

## Where GALILEO is weaker / needs to improve

- GALILEO currently abstracts away speech timing/overlap and turn-taking, which is a major real-world multi-turn failure source for voice agents.
- GALILEO could incorporate more explicit “session-level” degradation analyses (e.g., time-binned trajectories) as a presentation style, even in text-only settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work positioning: cite FDB-v2 as evidence that multi-turn performance degrades over time and that interaction protocol choices (pacing) can change trajectories; emphasize GALILEO’s complementary focus on ground-truth correctness under pressure.
- [ ] Writing idea: borrow the “trajectory over time” framing (their TT/IF curves) to motivate survival curves and show why multi-turn evaluation needs temporal dynamics, not just aggregate scores.

## Quotes / details to potentially cite

- Abstract (problem framing): “...their consistency and task performance in multi-turn settings remain underexplored.”
- Abstract (contribution): “...a streaming framework that integrates with an automated examiner that enforces staged goals under two pacing setups (Fast vs. Slow).”
- Limitation worth noting: “Reliance on an automated partner and transcript-based LLM scoring introduces prompt sensitivity, model biases, and calibration drift.”
