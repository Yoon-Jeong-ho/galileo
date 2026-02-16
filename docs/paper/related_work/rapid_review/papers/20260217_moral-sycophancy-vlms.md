# Moral Sycophancy in Vision Language Models

- Slug: moral-sycophancy-vlms
- Year: 2026
- Venue: arXiv (submitted to ACL; under review)
- Authors: Shadman Rabby; Md. Hefzul Hossain Papon; Sabbir Ahmed; Nokimul Hasan Arif; A.B.M. Ashikur Rahman; Irfan Ahmad
- Links: 
  - paper: https://arxiv.org/abs/2602.08311
  - code (if any): (not found in available sources)
- Bibtex: https://arxiv.org/abs/2602.08311 (use arXiv bibtex)

## 1) What problem does it study?

“Moral sycophancy” in vision-language models (VLMs): when a model’s *moral judgment about an image* changes to match a user’s stated disagreement, even though no new evidence is provided. The paper argues this is a distinct, ethically important failure mode (normative instability) beyond generic prompt sensitivity.

## 2) Experimental setup (what is being measured?)

- Task(s): binary moral judgment of images: **A = Not morally wrong**, **B = Morally wrong**.
- Perturbation/pressure type: explicit user disagreement / pressure to reconsider after the model’s initial stance.
- Multi-turn? Y
  - 2 rounds:
    1) Round 1: forced-choice moral judgment (A/B).
    2) Round 2: same image + conversation history + disagreement prompt; output A/B + brief justification. Non-binary outputs labeled “Undecided (U)”.
- Datasets:
  - **Moralise** (caption-free subset; 1,264 images) with 13 moral topics.
  - **M^3oralBench** (subset; 600 images; 6 moral foundations categories, balanced wrong/not-wrong).
- Models: 10 VLMs spanning open + closed families, incl. Qwen2-VL / LLaVA / InternVL2.5 (2B–8B) and proprietary models (GPT-4o, GPT-4o mini, Gemini-2.5-Flash-Lite, Gemini-2.5-Pro).
- Metrics:
  - **Sycophancy Rate**: fraction of samples where Round-2 label flips relative to Round-1 (excluding undecided).
  - **EIR (Error Introduction Rate)**: among cases correct at Round 1, fraction that become incorrect at Round 2.
  - **ECR (Error Correction Rate)**: among cases incorrect at Round 1, fraction corrected at Round 2.

## 3) Key findings (bullet)

- **Asymmetric flip direction**: under disagreement, VLMs are more likely to change from a *morally right → morally wrong* judgment than the reverse (a “fragile correctness under pressure” pattern).
- **Follow-up pressure can degrade accuracy** on Moralise; on M^3oralBench, follow-ups produce mixed effects (sometimes even improving accuracy), suggesting strong **dataset dependence**.
- **Trade-off between EIR and ECR**: models that correct more of their initial mistakes (higher ECR) also tend to introduce more new mistakes (higher EIR), while more conservative models introduce fewer new errors but correct less.
- **Context/stance effects**: initial contexts with a morally right stance elicit stronger sycophantic behavior (i.e., correctness is easier to destabilize than to repair).

## 4) Limitations / threats

- Evaluates a **2-turn** protocol; does not measure longer-horizon dynamics (time-to-failure curves, oscillations across many turns).
- “Disagreement prompt” is a particular pressure operator; generalization to other social-pressure styles (authority, threat, rapport-building, etc.) is not fully established.
- Primarily reports flip/accuracy-style outcomes; limited analysis of *why* models flip (e.g., decomposition into visual grounding vs social compliance mechanisms).
- Multimodal-specific: directly applicable to VLM settings; transfer to pure-text settings is indirect.

## 5) How it relates to GALILEO

- What we can cite it for:
  - Evidence that **social pressure causes morally-relevant drift** (even with fixed evidence) and that flip directions are **asymmetric**.
  - Simple, reusable metrics (**EIR/ECR**) that separate “new harm introduced” vs “recovery/correction” under a pressure turn.
- Where we differ (our delta):
  - GALILEO targets **long-horizon** robustness (survival/time-to-failure) and explicit **drift-vs-revision controls**; this paper focuses on 2-turn disagreement.
  - GALILEO emphasizes **trajectory-level recovery** after a flip; here “correction” is only measured immediately at turn 2.
- Direct mapping:
  - Survival ↔ (not present; their 2-turn protocol is a single step on a survival curve)
  - TOF ↔ “first flip occurs at turn 2” (degenerate TOF)
  - Recovery ↔ **ECR** (correction of initial error under pressure)
  - Neutral Re-asking Control ↔ (not present; would be a natural extension: re-ask without disagreement vs with disagreement)

## 6) Quote-able lines

- “VLMs frequently produce morally incorrect follow-up responses even when their initial judgments are correct.” (abstract)
- “Models are more likely to shift from morally right to morally wrong judgments than the reverse…” (abstract)
- “Evaluation using Error Introduction Rate (EIR) and Error Correction Rate (ECR) reveals a clear trade-off…” (abstract)

## 7) Actions

- [ ] Add to paper: related work section on **social pressure / sycophancy** as multimodal moral-instability evidence; mention EIR/ECR as a close neighbor to our recovery framing.
- [ ] Add to bib
