# Multi-Turn Puzzles: Evaluating Interactive Reasoning and Strategic Dialogue in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Jonathan Simon; Arian Hosseini; Sara Marie Mc Carthy; Tsendsuren Munkhdalai; Abhimanyu Goyal; Tomáš Kočiský; Shyam Upadhyay; Bahare Fatemi; Mehran Kazemi
- URL: https://arxiv.org/abs/2508.10142
- BibTeX key (if we add it): simon2025multiturnpuzzles
- Tags: multi-turn, interactive, reasoning, benchmark, information-seeking, planning

## One-sentence takeaway

MTP is a deterministic, rule-based benchmark of five interactive multi-turn “puzzle” environments that stress-test LLMs’ information-seeking, planning, and cross-turn logical consistency—useful as a protocol neighbor for GALILEO’s multi-turn dynamics (though not specifically social-pressure/sycophancy).

## What problem does it solve?

- Existing evaluation overweights single-turn, fully-specified prompts, under-measuring realistic settings where the model must **ask questions**, reason with **incomplete information**, and remain **logically consistent across turns**.
- Human evaluation / LLM-judge evaluation is expensive and noisy; they want **deterministic scoring**.

## What is the core method / protocol?

- Introduces **Multi-Turn Puzzles (MTP)**: five synthetic, rule-based interactive tasks with deterministic environments + scorers (no human/LLM judging).
- Evaluates frontier chat models in a fixed interaction loop: model ↔ environment for N turns, then a final answer/action is scored.
- Tasks (from the paper’s intro/table):
  - **Word Guess**: deduce a secret word with feedback; goal is minimal attempts.
  - **Movie Recommendation**: ask questions to infer a simulated user preference function; then recommend from a candidate set.
  - **Circuit Decoding**: probe unknown boolean circuits by querying inputs/outputs; then predict truth tables.
  - **Word Chaining**: alternating word game with an allow-list; goal is to avoid illegal moves and reach a terminal state.
  - **Twenty Questions**: model chooses a secret word and must answer user yes/no questions **without contradictions**.
- Public dataset link is provided in the HTML (HuggingFace dataset: `arianhosseini/mt_puzzles`).

## What are the key metrics?

All metrics are **task-specific and normalized** (per the paper’s task table):

- Word Guess: normalized number of attempts.
- Movie Recommendation: normalized rank of final recommendation.
- Circuit Decoding: normalized circuit-wise accuracy.
- Word Chaining: normalized % of trajectories ending successfully.
- Twenty Questions: normalized % of logically consistent trajectories.

## What are the main results?

- Across frontier models, **performance varies sharply by task**: they report relatively strong performance on Twenty Questions but “significant headroom” on the other tasks.
- Qualitative failure modes emphasized: **instruction-following failures**, **reasoning errors**, and **poor planning** (rather than mere knowledge gaps).

(Kept high-level here; the paper contains detailed plots/tables for specific models and task breakdowns.)

## How is this similar to GALILEO?

- Shared focus on **multi-turn dynamics** and the idea that single-turn accuracy can mask interactive failure modes.
- Uses **trajectory-level** scoring (episode success/consistency) rather than only final-turn correctness.
- Emphasizes “when/why models fail over turns” (planning, instruction-following, consistency), aligning with GALILEO’s multi-turn robustness framing.

## How is this different from GALILEO?

- Not centered on **social pressure / persuasion / sycophancy**; the adversary is the environment structure, not a persuasive user.
- No explicit **drift vs evidence-driven revision** control conditions.
- Metrics are mostly “end-to-end task success/consistency,” not explicitly **time-to-failure / flip / recovery** (though Word Guess implicitly measures attempts).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit pressure operators + controls (neutral vs pressure vs evidence), it provides a clearer lens on **belief drift mechanisms** than generic interactive tasks.
- If GALILEO tracks flip/recovery trajectories, it can provide richer temporal diagnostics than a single trajectory success score.

## Where GALILEO is weaker / needs to improve

- GALILEO could benefit from including at least one **fully deterministic, rule-based interactive environment** slice (judge-free), to reduce concerns about evaluator artifacts.
- If GALILEO focuses heavily on persuasion-style dialogue, it may under-cover “agentic” information-seeking/planning failures that MTP exposes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **judge-free** evaluation slice (rule-based environment + deterministic score) as a robustness control.
- [ ] Add a writing comparison: “GALILEO focuses on *pressure-induced belief drift*; MTP focuses on *interactive reasoning/planning under incomplete information*—complementary multi-turn failure modes.”
- [ ] If we already report time-to-failure, note that MTP provides a complementary family of **interactive task** benchmarks to validate generality beyond persuasion scenarios.

## Quotes / details to potentially cite

- Motivation (abstract): models “often struggle with nuanced environments or interactive tasks… need… logically consistent multi-turn dialogue, seek information and reason with incomplete data.”
- Benchmark design claim (abstract): “deterministic scoring mechanisms, thus eliminating the need for human intervention.”
- Task list (intro): Word Guess, Movie Recommendation, Circuit Decoding, Word Chain, Twenty Questions.
