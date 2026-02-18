# From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness

- Year: 2026
- Venue: AAAI 2026 TrustAgent Workshop (arXiv)
- Authors: Linbo Cao et al. (author list not captured in rapid fetch)
- URL: https://arxiv.org/abs/2602.12285
- BibTeX key (if we add it): Cao2026RoleAssignmentAgents (suggested)
- Tags: agents, persona, role-assignment, robustness, bias, prompt-injection

## One-sentence takeaway

Demographic persona/role prefixes—despite being task-irrelevant—can significantly and inconsistently degrade LLM-agent benchmark performance (reported up to 26.2%), implying persona conditioning is a robustness vulnerability for agentic systems.

## What problem does it solve?

- Identifies and quantifies a largely overlooked robustness failure mode in *LLM agents*: adding a demographic persona (gender/race/religion/profession) can change action-taking performance, not just tone or text.
- Motivates robustness testing for agent deployments where users (or adversaries) can induce personas/roles via prompt prefixes.

## What is the core method / protocol?

- Controlled *persona prefixing* experiment:
  - Prepend a fixed two-turn conversational prefix: user assigns persona ("From now on, you are a [ROLE] ..."), assistant acknowledges.
  - Keep the downstream task prompt/instructions unchanged.
  - Compare performance vs baseline (no persona prefix).
- Personas:
  - 23 personas spanning gender, race/origin, religion, profession.
- Benchmarks/domains (agentic, multi-step):
  - ALFWorld (household embodied-like planning; success rate)
  - WebShop (e-commerce interaction; reward)
  - Card Game (strategic reasoning; win rate/score)
  - OS Interaction (execute correct shell commands; accuracy)
  - Database (SQL correctness)
- Models:
  - GPT-4o-mini, DeepSeek-V3, Qwen3-235B.
- Deterministic decoding.

## What are the key metrics?

- Task success / accuracy / reward depending on benchmark (as above).
- Reported as relative change vs baseline (percentage increase/decrease).

## What are the main results?

- Persona assignments change agent performance across tasks and models.
- Largest reported degradations occur in higher-level reasoning/planning settings:
  - Up to 26.2% drop (notably in Card Game for DeepSeek-V3, per the paper).
  - ALFWorld success rate shifts up to ~14% (direction depends on persona/model).
- Technical tasks (OS Interaction, Database) are comparatively more stable (often within ~2–5% fluctuations).
- Effects vary by persona category and model; some personas sometimes *improve* performance, implying spurious correlations between persona cues and perceived competence.

## How is this similar to GALILEO?

- Both are about *multi-turn / multi-step* robustness failures driven by conversational context rather than task content.
- The paper’s persona-prefix manipulation is a concrete example of "contextual pressure" / framing that changes behavior, aligning with concerns about conversational drift, user conditioning, and stability.

## How is this different from GALILEO?

- Focuses on *demographic persona role assignment* (bias + volatility) rather than (primarily) belief revision / sycophancy-to-user-pressure trajectories.
- Evaluation is on agentic benchmarks (tool/action sequences) rather than dialogue-only stance flipping metrics.
- Doesn’t propose a GALILEO-style diagnostic metric (e.g., turn-to-flip / trajectory-level measures); it is a case study demonstrating effect sizes.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer causal manipulations of conversational pressure and trajectory-level metrics, it may offer a more directly measurable protocol for robustness characterization than broad persona sets.

## Where GALILEO is weaker / needs to improve

- If GALILEO targets only dialogue settings, this paper is a reminder to demonstrate transfer to *agentic* settings (planning + tool use) where consequences are operational.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph positioning persona/role-prefixing as a robustness threat model adjacent to social pressure/sycophancy.
- [ ] Consider an ablation: prepend demographic/profession persona prefixes to GALILEO tasks and measure stability/flip rates ("persona-induced drift").
- [ ] If GALILEO is agent-focused, include at least one agentic benchmark slice to show relevance beyond text-only dialogue.

## Quotes / details to potentially cite

- "Evaluating widely deployed models on agentic benchmarks spanning strategic reasoning, planning, and technical operations, we uncover substantial performance variations – up to 26.2% degradation, driven by task-irrelevant persona cues." (Abstract)
- Benchmarks used: ALFWorld, WebShop, Card Game, OS Interaction, Database. (Methodology)
