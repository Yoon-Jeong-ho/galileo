# SPILLage: Agentic Oversharing on the Web

- Year: 2026
- Venue: arXiv
- Authors: Jaechul Roh; Eugene Bagdasarian; Hamed Haddadi; Ali Shahin Shamsabadi
- URL: https://arxiv.org/abs/2602.13516
- BibTeX key (if we add it): Roh2026SPILLage
- Tags: web-agents, privacy, oversharing, behavioral-traces, evaluation, taxonomy

## One-sentence takeaway

Web agents overshare users’ task-irrelevant attributes not only via typed text but (more dominantly) via observable behaviors (click/scroll/navigation), and simple input sanitization can improve both privacy and task success.

## What problem does it solve?

- Defines and measures *natural* (non-adversarial) privacy leakage by web agents acting on live sites when prompts include mixed task-relevant and task-irrelevant user attributes.
- Highlights a blind spot in prior “leakage” evaluations that focus mostly on text outputs and adversarial prompt injection, missing what websites can infer from action traces.

## What is the core method / protocol?

- Formalizes **Natural Agentic Oversharing**: unintentional disclosure of task-irrelevant user information through an agent’s web action trace.
- Introduces **SPILLage** taxonomy with two axes:
  - Channel: **content** (typed inputs) vs **behavior** (click/scroll/navigation).
  - Directness: **explicit** (verbatim) vs **implicit** (inferable).
  - Yields 4 quadrants: Explicit Content, Implicit Content, Explicit Behavioral, Implicit Behavioral.
- Constructs a benchmark of **180 tasks** on live e-commerce sites (Amazon, eBay) with **ground-truth partitioning** of attributes into task-relevant vs task-irrelevant.
- Runs 1,080 agent executions across:
  - Frameworks: Browser-Use; AutoGen
  - Backbones: OpenAI gpt-4o, o3, o4-mini
- Uses a step-level **LLM-judge audit** that inspects each action (and associated context) to label oversharing events and categorize them in the taxonomy.
- Defense probe: (i) prompt-level instruction “avoid irrelevant info” (can backfire), (ii) **remove task-irrelevant information before execution** (sanitization), measuring both privacy and task success impact.

## What are the key metrics?

- Oversharing **occurrences** (count of oversharing events).
- **Oversharing rate**: occurrences divided by total actions (can exceed 1 if one action reveals multiple attributes).
- **Task success** (framework-specific; includes LLM-judge for AutoGen).

## What are the main results?

- Oversharing is pervasive across models/frameworks/sites.
- **Behavioral oversharing dominates content oversharing** (reported ~5x overall).
- Framework trade-off: fine-grained action frameworks can increase total oversharing occurrences (longer traces), while higher-level frameworks can concentrate risk per step (higher per-step oversharing rate).
- Prompt-level mitigation (“be careful not to use irrelevant info”) can **increase** oversharing rate in their pilot.
- **Input sanitization** (removing task-irrelevant attributes before agent execution) can **improve task success** (reported up to +17.9%) while also reducing oversharing.

## How is this similar to GALILEO?

- Shares the core concern that *agentic systems* create new leakage surfaces beyond plain text outputs.
- Reinforces the need to reason about what external observers can learn from interaction traces (a contextual-integrity framing).
- Emphasizes measurement via real task trajectories rather than only static text-generation tests.

## How is this different from GALILEO?

- Focuses specifically on **web agents** and leakage via **UI behaviors** (clicks/scrolls) as an observable channel.
- Uses e-commerce tasks and a step-level oversharing taxonomy; does not appear to target GALILEO’s specific domain/mechanisms (as framed in our paper) but is closely adjacent as “agentic privacy evaluation.”

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides formal guarantees, mechanism-level protections, or broader threat models beyond passive web observers, SPILLage is primarily an empirical taxonomy + benchmark.
- SPILLage’s main “defense” finding is input sanitization; if GALILEO offers principled runtime controls or auditing beyond sanitization, that would be a differentiator.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently emphasizes textual disclosure, SPILLage suggests we should explicitly cover **behavioral channels** and inference from action patterns.
- Consider evaluating “mitigations that backfire” (instruction-based guardrails increasing salience / longer traces).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph positioning “oversharing” as a broader lens than “prompt injection leakage,” including behavioral traces.
- [ ] Consider a taxonomy axis in GALILEO writeup: *what* is exposed (content vs behavior/metadata) and *how* (explicit vs implicit).
- [ ] If we have an agent benchmark, add an ablation: **sanitize task-irrelevant context** and measure both success and leakage.
- [ ] If we propose prompt-based mitigations, include a check for **backfire** (increased leakage due to longer trajectories or salience).

## Quotes / details to potentially cite

- “Natural Agentic Oversharing—the unintentional disclosure of task-irrelevant user information through an agent trace of actions on the web.”
- “Behavioral oversharing dominates content oversharing” (reported ~5×).
- Sanitization claim: removing task-irrelevant info before execution improves task success “by up to 17.9%.”
