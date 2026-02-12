# Chain-of-Verification Reduces Hallucination in Large Language Models

- Slug: cove_2024
- Year: 2024
- Venue: Findings of ACL 2024
- Authors: Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, Jason Weston
- Links:
  - paper: https://aclanthology.org/2024.findings-acl.212/
  - pdf: https://aclanthology.org/2024.findings-acl.212.pdf
- Bibtex: `dhuliawala-etal-2024-chain` (see `references.bib`)

## 1) What problem does it study?
LLM hallucination reduction via *self-deliberation*: draft an answer, then explicitly verify/fact-check it, then revise.

## 2) Experimental setup (what is being measured?)
- Task(s): list questions (Wikidata), closed-book MultiSpanQA, long-form generation
- Perturbation/pressure type: none (focus is factuality / hallucination)
- Multi-turn? Y (structured multi-step prompting pipeline: draft → plan verification questions → answer them independently → revise)
- Metrics: hallucination / factuality metrics as defined per task (paper-specific)

## 3) Key findings (bullet)
- A structured verification chain (CoVe) reduces hallucinations across multiple tasks compared to direct answering.
- Key design choice: answer verification questions *independently* to avoid copying the initial (possibly wrong) draft.

## 4) Limitations / threats
- Not a multi-turn *dialogue* robustness setting; it is a single query with a multi-step internal procedure.
- The gains depend on the model’s ability to generate good verification questions and to answer them faithfully.

## 5) How it relates to GALILEO
- What we can cite it for:
  - Prior work showing that “verify-then-answer” style procedures can measurably improve truthfulness / reduce hallucination.
- Where we differ (our delta):
  - GALILEO targets *multi-turn conversational pressure* (e.g., user steering) and evaluates failure dynamics (Survival/TOF) plus *Recovery* in dialogue.
- Direct mapping:
  - Survival ↔ not directly (they do not model time-to-failure across turns)
  - TOF ↔ not directly
  - Recovery ↔ conceptually adjacent: revise-after-verification is a form of answer repair
  - Neutral Re-asking Control ↔ not present

## 6) Quote-able lines
- CoVe pipeline: draft → plan verification questions → answer independently → generate final verified response.

## 7) Actions
- [ ] Add to paper: Related Work / Truthfulness & self-verification (1 sentence contrasting with our multi-turn + recovery evaluation)
- [x] Add to bib
