# Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs

- Year: 2026
- Venue: arXiv
- Authors: Yiyang Lu; Kai Chen; Ruigang Liang; Jinwen He; Yue Zhao
- URL: https://arxiv.org/abs/2601.14340
- BibTeX key (if we add it): lu2026tst
- Tags: multi-turn, llm-security, backdoor, structural-trigger, supply-chain

## One-sentence takeaway

A supply-chain backdoor can be triggered purely by *conversation structure* (e.g., the k-th turn) rather than any user-visible text, achieving ~99% attack success and bypassing prompt-centric defenses.

## What problem does it solve?

- Shows a new backdoor trigger channel for chat LLMs: system-inserted structure (role tags / separators / turn index), which typical audits and defenses ignore.
- Models a realistic threat: poisoned open-source/instruction-tuned checkpoints that look normal in routine tests but misbehave at a specific turn during real multi-turn usage.

## What is the core method / protocol?

- **Turn-based Structural Trigger (TST):** the trigger is the dialogue *round index* (e.g., “activate on the k-th assistant reply”), independent of the user’s prompt tokens.
- Implantation via supervised fine-tuning with a small set of **poisoned multi-turn dialogues**:
  - Clean behavior on non-target turns.
  - On the target turn, force an attacker-chosen **payload** (e.g., unsolicited ad insertion / harmful content).
- Key design idea: the model can condition on structural formatting implicitly present in chat templates (role tags, separators), so “turn position” becomes a stable activation signal.

## What are the key metrics?

- **ASR (Attack Success Rate)** on the triggered turn.
- **Utility / clean performance retention** vs the clean model (reported as percent retained).
- Robustness of ASR under representative **defenses** (prompt sanitization / paraphrasing style defenses are highlighted as insufficient for structure triggers).
- Cross-dataset generalization of ASR across instruction datasets.

## What are the main results?

- Across 4 open-source LLM families:
  - Average **ASR ~99.5%** with minimal utility degradation.
  - Under 5 representative defenses: average **ASR ~98%**.
  - Cross-instruction-dataset generalization: average **ASR ~99.2%**.
- Poisoning budget mentioned: **~1,800 poisoned dialogues** used for injection (as reported in the paper’s introduction).

## How is this similar to GALILEO?

- Both care about **multi-turn behavior** and failure modes that only surface after several dialogue rounds.
- Reinforces that *format/structure* (not just user text) can dominate outcomes—relevant when GALILEO designs evaluation protocols for drift/instability across turns.

## How is this different from GALILEO?

- This is primarily **adversarial security / backdoor** work, not measuring naturalistic instability, belief drift, or recovery.
- Trigger is deterministic and attacker-inserted; GALILEO’s focus is more on evaluation/robustness under interaction dynamics (non-poisoned models) and causal controls.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can frame multi-turn failures with **control conditions** (evidence vs drift) and richer *behavioral* metrics, whereas ASR is a narrow security metric.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluation only varies user-visible prompts, it may miss **system-template / structural** confounds (role-tagging, turn counters, truncation policy, system prompt placement).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit “**structure sensitivity**” checklist to multi-turn evaluations: vary chat template details (role tokens/separators, message packing, turn indexing) while holding user-visible text constant.
- [ ] Consider a small ablation in the paper appendix: demonstrate whether key multi-turn metrics are invariant to common template variations (or quantify variance).
- [ ] In related-work writing: cite this as evidence that **turn index** and structural cues are an under-audited axis in multi-turn systems.

## Quotes / details to potentially cite

- “Existing backdoor attacks and defenses are largely prompt-centric… overlooking structural signals in multi-turn conversations.” (Abstract)
- “TST… activates from dialogue structure, using the turn index as the trigger and remaining independent of user inputs.” (Abstract)
- Reported summary numbers: ASR 99.52% (avg), ASR under defenses 98.04% (avg), cross-dataset ASR 99.19% (avg). (Abstract)
