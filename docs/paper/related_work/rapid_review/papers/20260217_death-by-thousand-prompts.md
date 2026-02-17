# Death by a Thousand Prompts: Open Model Vulnerability Analysis

- Year: 2025
- Venue: arXiv (cs.CR, cs.LG)
- Authors: Amy Chang, Nicholas Conley, Harish Santhanalakshmi Ganesan, Adam Swanda (Cisco AI Threat Research & Security)
- URL: https://arxiv.org/abs/2511.03247
- BibTeX key (if we add it): chang2025death
- Tags: prompt-injection, jailbreak, multi-turn, red-teaming, open-weights, evaluation

## One-sentence takeaway

Automated red-teaming of eight open-weight LLMs finds multi-turn prompt-injection/jailbreak attacks are dramatically more successful (2–10×) than single-turn, with large variation by model/alignment posture.

## What problem does it solve?

- Provides comparative evidence about how vulnerable popular open-weight models are to prompt injection and jailbreaks, especially in multi-turn settings that better resemble real deployments.
- Helps practitioners reason about risk when choosing a base model for fine-tuning/deployment.

## What is the core method / protocol?

- Black-box, automated adversarial testing (“AI Validation” platform) over 8 open-weight instruction-tuned LLMs.
- Evaluates both:
  - Single-turn attacks
  - Multi-turn attacks (iterative steering across turns)
- Success judged with an LLM-as-judge (paper explicitly notes judge variability and replication caveats).
- Models covered (as listed in the paper’s executive summary): Qwen3-32B, DeepSeek v3.1, Gemma 3-1B-IT, Llama 3.3-70B-Instruct, Phi-4, Mistral Large-2, GPT-OSS-20b, GLM 4.5-Air.

## What are the key metrics?

- Attack Success Rate (ASR) for single-turn vs multi-turn jailbreak/prompt-injection strategies.
- “Security gap” = difference in ASR between multi-turn and single-turn (reported as a positive gap for most models).

## What are the main results?

- Multi-turn attacks succeed far more often than single-turn:
  - Multi-turn ASR range: 25.86% to 92.78% across tested models.
  - Reported as 2× to 10× increase vs single-turn baselines.
- Claim/interpretation: capability-focused models (e.g., Llama 3.3, Qwen 3) show higher multi-turn susceptibility, while more safety-oriented designs (e.g., Gemma 3) appear more balanced.
- Threat-category patterning: manipulation/misinformation/malicious code generation show high multi-turn success; “top subthreats” concentrate risk.

## How is this similar to GALILEO?

- If GALILEO targets safe/robust long-horizon interaction (multi-step dialogs, agentic workflows, tool-use), this paper reinforces that *multi-turn* is the critical regime where guardrails often fail.
- Emphasizes evaluation methodology and reporting that can inform GALILEO’s related-work narrative (single-turn vs multi-turn; black-box red-teaming; ASR + gap metrics).

## How is this different from GALILEO?

- This is primarily a vulnerability measurement/reporting effort, not a new defense/guardrail method.
- Uses an LLM-as-judge evaluation and a proprietary testing platform; may not provide fully reproducible open benchmarks/attack sets.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a concrete defense or training protocol, it can claim contribution beyond measurement: improving robustness in the multi-turn regime rather than only quantifying failures.
- If GALILEO uses deterministic or human-verified evaluation, it may avoid some LLM-judge variability caveats.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluation is mostly single-turn, this paper suggests it may substantially underestimate real risk.
- If GALILEO does not report multi-turn “gap” style metrics, it may be harder to compare robustness across interaction lengths.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/expand a multi-turn red-teaming evaluation section; report single-turn vs multi-turn ASR and the gap.
- [ ] Include threat-category breakdowns (e.g., manipulation/misinformation/malicious code) to show where GALILEO helps most.
- [ ] In related work, explicitly cite the “multi-turn is 2–10× worse” framing as motivation for long-horizon defenses.

## Quotes / details to potentially cite

- “multi-turn attacks achieving success rates between 25.86% and 92.78% — representing a 2× to 10× increase over single-turn baselines.” (abstract)
- “Multi-turn Attacks Remain the Primary Failure Mode … ranging from 25.86 percent … to 92.78 percent … representing up to a 10x increase over single-turn baselines.” (Findings)
