# LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge

- Year: 2025
- Venue: arXiv (challenge featured at SaTML 2025)
- Authors: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai‑Chieh Liao, Chi‑Huang Liu, Chun‑Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun‑Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin
- URL: https://arxiv.org/abs/2506.09956
- BibTeX key (if we add it): Abdelnabi2025LLMailInject
- Tags: prompt-injection, indirect-prompt-injection, benchmark, dataset, tool-calling, email-agent, adaptive-adversary

## One-sentence takeaway

LLMail-Inject releases a large-scale dataset (208k adaptive submissions) from a realistic email-agent prompt-injection competition, enabling systematic evaluation of defenses against adaptive adversaries in tool-using LLM systems.

## What problem does it solve?

- Existing prompt-injection defenses/benchmarks often lack *adaptive* attackers and realistic end-to-end setups (retrieval + tool calling + detection), making it hard to compare defenses under realistic threat models.
- Specifically targets the “instruction–data separation” failure mode in LLM applications that ingest untrusted data (emails/search results) and can call tools.

## What is the core method / protocol?

- Public challenge (“Adaptive Prompt Injection Challenge”, LLMail-Inject) with a simulated LLM-based email assistant.
- Participants craft a single attacker email intended to be retrieved and then cause an *unauthorized* tool call (send_email) despite defenses.
- Multiple “levels” vary retrieval setup and difficulty:
  - L1/L2: summarization of recent emails (attacker email always included).
  - L3/L4: keyword-based retrieval over a mailbox; L4 additionally requires exfiltrating a secret value via the tool-call argument.
- Evaluates multiple defenses (and their combination), including:
  - “Spotlighting” / data marking with randomized delimiters.
  - Black-box prompt-injection classifier (Prompt Shield).
  - LLM-as-judge detector.
  - Internal-state / task-drift detector (TaskTracker; probe on activation deltas).
- Uses two LLM backends representative of (i) open-source long-context (Phi-3-medium-128k-instruct; tool calling via JSON prompting/parsing) and (ii) a model trained with instruction hierarchy (GPT-4o-mini; emails passed as tool messages; native tool calling).

## What are the key metrics?

- Empirical attack success rate under each defense / retrieval level (challenge scoring), under *adaptive* knowledge of defenses.
- Dataset scale and diversity: number of unique submissions and participants.
- (Implied) false positive rate calibration: defense thresholds tuned to keep FPR < 5% on separate test sets.

## What are the main results?

- Produces a large benchmark dataset: 208,095 unique attack submissions from 839 participants.
- Demonstrates a practical, end-to-end evaluation setting where a successful attack must pass retrieval, evade detectors, and induce a correct tool call.
- Provides evidence/analysis intended to yield insights about instruction–data separation failures and why practical defenses remain brittle under adaptive pressure.

## How is this similar to GALILEO?

- Same core security problem space: untrusted retrieved content attempting to override intended task and trigger unintended tool actions.
- Emphasizes structural failures of “instruction vs data” separation and evaluates defenses in a tool-using pipeline.

## How is this different from GALILEO?

- Focuses on *email-agent* scenario and a competition-derived dataset; GALILEO may target a different application domain / protocol.
- Their primary contribution is benchmark + analysis rather than a new defense mechanism (beyond describing the challenge defenses).
- Defenses include black-box and proprietary components (e.g., Prompt Shield), which may be less reproducible.

## Where GALILEO is stronger / cleaner (if true)

- Potentially more controllable/reproducible evaluation if GALILEO avoids proprietary classifiers and defines a fully open threat model + artifacts.
- If GALILEO provides formal guarantees or tighter protocol-level separation, it could go beyond the “prompting + detectors” mix evaluated here.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks large-scale adaptive-adversary evaluation, LLMail-Inject highlights the importance of testing against attackers who know the defense.
- If GALILEO does not include realistic retrieval + tool-calling end-to-end benchmarks, this dataset could expose gaps.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an evaluation slice inspired by LLMail-Inject: end-to-end (retrieval → model → tool call), with attackers adaptive to the defense.
- [ ] Cite LLMail-Inject as evidence that adaptive attacker evaluation changes conclusions about defense robustness.
- [ ] If applicable, compare against their “spotlighting” randomized delimiters and task-drift detection framing.

## Quotes / details to potentially cite

- “...participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant.”
- “...dataset of 208,095 unique attack submissions from 839 participants.”
- Level design includes retrieval variation and an exfiltration objective via tool-call arguments (L4).
