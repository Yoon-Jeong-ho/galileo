# Understanding the Impact of Confidence in Retrieval Augmented Generation: A Case Study in the Medical Domain

- Year: 2025
- Venue: BioNLP 2025 (workshop, colocated with ACL 2025) (per arXiv comments)
- Authors: Shintaro Ozaki, Yuta Kato, Siyuan Feng, Masayo Tomita, Kazuki Hayashi, Wataru Hashimoto, Ryoma Obara, Masafumi Oyamada, Katsuhiko Hayashi, Hidetaka Kamigaito, Taro Watanabe
- URL: https://arxiv.org/abs/2412.20309
- BibTeX key (if we add it): ozaki2025confidence-rag-med
- Tags: rag, confidence, calibration, uncertainty, medical, qa

## One-sentence takeaway

RAG-style document insertion can change *probability-derived confidence* and calibration in medical QA, and some models (notably Phi-3.5 and Qwen2.5 in this study) can internally distinguish relevant vs irrelevant inserted passages as reflected in their output probabilities.

## What problem does it solve?

- In high-stakes domains (medicine), people use RAG to improve factuality, but it is unclear how RAG affects *model confidence* (as measured by output probabilities) and whether models become overconfident or better calibrated.
- Provides a controlled evaluation protocol to test whether inserting relevant/irrelevant “retrieved” text changes confidence, entropy, accuracy, and calibration error.

## What is the core method / protocol?

- Controlled “pseudo-RAG” setup for multiple-choice medical QA:
  - Datasets: PubMedQA (3-way: yes/no/maybe) and MedMCQA (4-way), restricted to instances with supporting evidence passages.
  - Instead of free-form generation + regex extraction, they use *force decoding / choice scoring*:
    - Compute log-probability for each candidate choice given the prompt, softmax-normalize to get a probability distribution over choices.
    - Take best probability (max over choices) as a confidence proxy.
- Insert documents (the dataset’s explanation passages) into the prompt in various *positions*:
  - Pre-Q (before question), Aft-Q (between question and choices), Aft-C (after choices).
  - Motivated by “lost in the middle” effects for long-context.
- Evaluate three document-content scenarios:
  - Ans1: only the answer-supporting explanation.
  - Ans1-Oth2: answer-supporting explanation + two irrelevant passages.
  - Oth3: three irrelevant passages.
- Models tested (9 total): Phi-3.5 (3.8B), Gemma2 (2B), PMC-Llama (13B), Llama2 (70B), Llama3.1 (8B/70B), Meditron (70B), Qwen2.5 (14B/72B).

## What are the key metrics?

- Accuracy on the multiple-choice QA.
- Entropy of the choice distribution (lower entropy = more peaked/confident distribution).
- Best probability (max choice probability) as a confidence score.
- Adaptive Calibration Error (ACE) as calibration error metric (preferred over ECE in their discussion for multi-class settings).

## What are the main results?

- For some models (highlighted: Phi and Qwen), inserting an answer-supporting passage tends to:
  - Increase accuracy and increase confidence / decrease entropy on correct cases.
  - Change calibration error (ACE) in ways that can diagnose model sensitivity to document quality.
- Inserting unrelated passages (Oth3) *rarely* improves confidence in a desirable way for Phi/Qwen, suggesting these models can detect irrelevance and avoid being “fooled” by arbitrary inserted text (at least under this protocol).
- Other model families (notably several Llama/Gemma configurations in their tables) show inconsistent or counterintuitive behaviors, suggesting some models may not effectively use (or may be disrupted by) inserted passages under these prompt templates.
- Prompt position effects:
  - They report mixed evidence for “lost in the middle”: for Phi/Qwen, entropy improvements sometimes favor Aft-C (document after choices), while accuracy may favor putting documents earlier.

## How is this similar to GALILEO?

- If GALILEO uses retrieval or evidence insertion, this paper provides a *diagnostic lens*:
  - Use probability-based confidence / entropy to test whether the generator actually uses retrieved evidence vs ignores it.
  - Evaluate robustness to irrelevant retrievals (does confidence drop / remain stable when retrieval is wrong?).
- Highlights that “RAG improves accuracy” is not sufficient; we should also monitor confidence and calibration, especially in high-stakes settings.

## How is this different from GALILEO?

- This work is primarily an *evaluation/analysis* of confidence behavior in a pseudo-RAG prompt setup (not proposing a new retrieval method).
- It focuses on multiple-choice QA with choice scoring (force decoding), which may differ from GALILEO’s target tasks (e.g., open-ended generation, structured outputs, planning).
- Uses explanation passages from QA datasets as inserted “documents,” rather than a full retrieval pipeline with real corpora, retriever errors, and end-to-end retrieval scoring.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes real retrieval + evidence selection, it can study more realistic failure modes (retriever noise, near-duplicates, outdated docs) beyond pseudo-RAG.
- If GALILEO already tracks uncertainty/calibration or implements confidence-aware control (e.g., retrieval triggers, abstention), it goes beyond the descriptive analysis here.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly evaluate calibration/confidence under retrieval perturbations, this paper suggests an actionable missing evaluation axis.
- If GALILEO relies only on accuracy/quality scores, it may miss “overconfident wrong” regimes that matter in high-stakes usage.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “pseudo-RAG” perturbation test suite: for a held-out set, insert (a) relevant evidence, (b) irrelevant evidence, (c) mixed evidence; measure changes in confidence/entropy vs correctness.
- [ ] Track calibration metrics (ACE/ECE variants) for retrieval-augmented vs retrieval-free settings to detect overconfidence.
- [ ] Test evidence placement / ordering as an ablation (front vs middle vs end) to quantify “lost in the middle” for GALILEO prompts.
- [ ] Consider using “choice scoring” style evaluation (or a proxy) for any multiple-choice subtask, to avoid brittle regex extraction when measuring confidence.

## Quotes / details to potentially cite

- They frame the main research question as whether RAG improves confidence for outputs, and note risk of overconfidence despite higher accuracy.
- They define confidence from output probabilities (best probability) and analyze entropy and ACE to study calibration under different document insertion scenarios.
