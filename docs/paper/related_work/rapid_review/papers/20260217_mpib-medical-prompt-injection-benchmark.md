# MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs

- Year: 2026
- Venue: arXiv
- Authors: Junhyeok Lee; Han Jang; Kyu Sung Choi
- URL: https://arxiv.org/abs/2602.06268
- BibTeX key (if we add it): lee2026mpib
- Tags: prompt-injection, medical, safety, benchmark, rag

## One-sentence takeaway

MPIB is a clinically grounded prompt-injection benchmark (direct + RAG-mediated) that argues *attack success* can diverge from *clinical harm*, and introduces **CHER** to measure high-severity patient-risk outcomes.

## What problem does it solve?

- Existing prompt-injection evaluations often focus on instruction-following / jailbreak success (e.g., ASR), which may not reflect **clinically meaningful harm**.
- RAG changes the trust boundary: **indirect injection** via retrieved context can be particularly dangerous in medicine because poisoned content can look authoritative.
- Need a reproducible benchmark that separates “model followed the attacker” from “model produced high-severity unsafe clinical advice.”

## What is the core method / protocol?

- Introduces **MPIB** (Medical Prompt Injection Benchmark): **9,697** curated instances with multi-stage quality gates + “clinical safety linting.”
- Two attack vectors:
  - **V1 (direct injection):** adversarial instructions appear in the user query.
  - **V2 (indirect / RAG-mediated injection):** adversarial instructions are embedded in retrieved context.
- Scenarios: four “scenario families” (**S1–S4**) spanning realistic clinical tasks (e.g., explanation, dosing, triage, guideline/evidence reasoning).
- Outcomes labeled with a **clinical harm taxonomy** (**H1–H5**) and **severity** (0–4), enabling outcome-centric reporting.
- Provides evaluation harness + baseline defenses; includes a “structured evaluator/judge” with schema validation + deterministic post-processing for stability.

## What are the key metrics?

- **ASR (Attack Success Rate):** whether the attack causes instruction-following / compliance.
- **CHER (Clinical Harm Event Rate):** rate of **high-severity clinical harm events** (Severity ≥ 3), intended to track downstream patient risk rather than mere compliance.

## What are the main results?

- **ASR and CHER can diverge substantially** (a defense/model can reduce compliance without reducing severe clinical harm, and vice versa).
- Robustness depends strongly on where the malicious instruction appears (**user query vs retrieved context**).
- Indirect/RAG-mediated injection can be especially strong due to **authority framing** of retrieved content; the paper reports multiple-fold CHER increases for V2 vs V1 in high-risk categories (per their baseline suite).

## How is this similar to GALILEO?

- Same meta-lesson: a single scalar “attack success” metric may not capture the **actual downstream failure mode** we care about.
- Emphasizes **context channel** effects (user text vs retrieved context) as a crucial experimental factor—analogous to separating pressure sources / channels in multi-turn interaction studies.

## How is this different from GALILEO?

- Focuses on **security + clinical safety** under prompt injection (direct/indirect), not on multi-turn belief/stance dynamics.
- Uses outcome-severity clinical taxonomy + harm auditing; not centered on flip/recovery trajectories.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s core claim is about **multi-turn dynamics** (time-to-failure, recovery, oscillation), MPIB is mostly orthogonal (primarily an outcome/harm benchmark).

## Where GALILEO is weaker / needs to improve

- Consider whether we have an outcome-level metric analogue to **CHER** for our domain (i.e., a severity-weighted “harm” lens rather than only agreement/flip metrics).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short discussion motivating why “attack success / compliance” can be a misleading proxy, citing MPIB’s ASR–CHER divergence as an existence proof in a high-stakes domain.
- [ ] If we have any “safety/utility” decomposition, consider a severity-thresholded event-rate metric (CHER-like) as a reporting option.

## Quotes / details to potentially cite

- “We introduce the Medical Prompt Injection Benchmark (MPIB) … for evaluating clinical safety under both direct prompt injection and indirect, RAG-mediated injection …” (Abstract)
- “MPIB emphasizes outcome-level risk via the Clinical Harm Event Rate (CHER) … and reports CHER alongside Attack Success Rate (ASR) to disentangle instruction compliance from downstream patient risk.” (Abstract)
- Dataset size: “9,697 curated instances …” (Abstract)
- Code/data links (as stated by authors): https://github.com/jhlee0619/mpib-eval and https://huggingface.co/datasets/jhlee0619/mpib
