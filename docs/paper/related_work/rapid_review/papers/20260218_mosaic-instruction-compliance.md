# Deconstructing Instruction-Following: A New Benchmark for Granular Evaluation of Large Language Model Instruction Compliance Abilities

- Year: 2026
- Venue: EACL 2026 (accepted)
- Authors: Alberto Purpura, Li Wang, Sahil Badyal, Eugenio Beaufrand, Adam Faulkner
- URL: https://arxiv.org/abs/2601.18554
- BibTeX key (if we add it): mosaic-purpura-2026
- Tags: instruction-following, compliance, constrained-generation, benchmark, evaluation, constraint-interactions

## One-sentence takeaway

MOSAIC is a large, dynamically-generated constrained text generation benchmark that decouples “task success” from “instruction compliance” and measures how compliance changes with constraint type, count, ordering, and interactions.

## What problem does it solve?

- Existing instruction-following benchmarks often (i) use toy constraints, (ii) entangle constraints with the underlying task, and/or (iii) evaluate only a few constraints at a time—making it hard to diagnose real compliance failures and interaction/position effects.
- Data leakage risk: static public prompts can be memorized; they propose dynamic generation.

## What is the core method / protocol?

- Build MOSAIC: dynamically generate prompts for constrained text generation by combining:
  - a content-generation task (e.g., marketing email, product review, FAQ entry, internal memo),
  - a product/service description (8 types),
  - a modular *list* of constraints.
- Constraints are “application-oriented” and span multiple groups (formatting, lexical, syntactic, semantic); paper claims 21 constraint types and up to ~20 constraints per prompt (avg ~10.5).
- Evaluate multiple LLMs (5 models across families) and report:
  - overall prompt-level compliance,
  - constraint-type breakdown,
  - pairwise interaction effects (synergy vs conflict),
  - ordering / position effects (primacy/recency biases).

## What are the key metrics?

- Prompt-level compliance (did the output satisfy all constraints).
- Single-constraint compliance rates by constraint type.
- Pairwise interaction analysis (how adding constraint B changes compliance on A, and vice versa).
- Position-based compliance analysis (constraint satisfaction vs its position in the list; primacy/recency).

## What are the main results?

- Instruction compliance is not monolithic: it varies substantially by constraint type, number of constraints, and their ordering.
- Models show distinct positional biases (primacy and recency effects), and non-trivial constraint interactions (some constraints reinforce each other; others conflict and reduce compliance).
- Compared to prior work, MOSAIC is larger and denser in constraints (Table 1: 4,000 prompts; ~10.5 constraints/prompt).

## How is this similar to GALILEO?

- Shared emphasis on *fine-grained evaluation* rather than a single aggregate score.
- Highlights that “control” (following structured requirements) degrades with complexity and interacts with ordering—relevant for any system relying on multiple simultaneous constraints.

## How is this different from GALILEO?

- MOSAIC is specifically about instruction/compliance evaluation for constrained generation prompts; it is not primarily about retrieval, tool-use, or downstream task performance.
- Uses synthetic prompt construction and constraint lists to isolate compliance; GALILEO may target different modalities/tasks/metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates on real user tasks or end-to-end pipelines, it may offer higher ecological validity than synthetic product/content prompts.
- If GALILEO has stronger ground-truth task correctness measures, it can complement MOSAIC’s compliance focus.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks systematic tests for constraint interactions and ordering effects, MOSAIC suggests these are essential diagnostics.
- If GALILEO doesn’t separate “task accuracy” from “meta-constraint compliance,” results may conflate two failure modes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “constraint-list stress test” slice: vary constraint count (e.g., 1→5→10→15) and report compliance curves.
- [ ] Add ordering experiments (shuffle constraints; measure primacy/recency) and report positional bias.
- [ ] Add pairwise interaction heatmaps for key constraint pairs used in GALILEO (identify conflicts that drive failures).
- [ ] In writing, explicitly distinguish task correctness vs compliance, and report both.

## Quotes / details to potentially cite

- MOSAIC frames a key distinction: task accuracy (correct answer) vs instruction following/compliance (adherence to meta-rules about format/style/structure).
- Scale/complexity claim (Table 1): 4,000 prompts, ~21 constraint types, and ~10.5 constraints per prompt; designed to probe interactions and ordering effects.
