# An Empirical Study of Group Conformity in Multi-Agent Systems

- Year: 2025
- Venue: Findings of ACL 2025 (per arXiv page)
- Authors: Min Choi; Keonwoo Kim; Sungwon Chae; Sangyeob Baek
- URL: https://arxiv.org/abs/2506.01332
- BibTeX key (if we add it): choi2025conformity-multiagent (suggested)
- Tags: conformity, multi-agent, debate, opinion-dynamics, bias-propagation

## One-sentence takeaway

In multi-agent LLM debates on contentious topics, a “neutral” evaluator agent systematically shifts toward the numerically dominant side and can be swayed even more by a single higher-capability model than by a larger group of weaker models.

## What problem does it solve?

- Characterizes *how* opinion/bias can emerge and propagate in LLM multi-agent interactions on contentious social issues (beyond protected-attribute bias).
- Empirically tests whether classic social-science conformity dynamics (majority influence, “spiral of silence”-like effects) appear in LLM-agent debates, and what factors (group size vs. model capability) drive them.

## What is the core method / protocol?

- Debate simulation with 3 roles:
  - Proponent agents argue “pro”
  - Opponent agents argue “con”
  - A Neutral agent is initialized as “centrist/neutral” and, after each turn, evaluates both sides and chooses the stance it finds more persuasive.
- Manipulations:
  - Group size: minority vs. majority counts of proponent/opponent agents.
  - “Intelligence” level: mix models from families like GPT / Claude / Qwen (paper frames larger models as more intelligent).
- Scale: >2,500 simulated debates across five contentious topics (example given: UBI necessity).
- Analysis: statistical analysis of how frequently and how strongly the neutral agent aligns with the majority vs. the higher-capability side.

## What are the key metrics?

- Conformity / alignment rate of the neutral agent with:
  - the numerically dominant group (majority effect), and
  - the higher-“intelligence” agent/group.
- “Extent” of stance adoption over time/turns (described qualitatively on the arXiv HTML page; details likely in later sections not captured in the truncated fetch).

## What are the main results?

- Clear majority effect: neutral agents tend to align with numerically dominant groups.
- Capability effect: a single high-capability agent can influence the neutral agent more effectively than a larger group of lower-capability agents.
- Takeaway risk: multi-agent discussions can amplify bias on contentious issues; “who” (capability) participates may matter as much as “how many.”

## How is this similar to GALILEO?

- Both touch opinion dynamics / interaction effects in multi-agent settings, and the risk that multi-agent discourse produces systematic shifts rather than independent aggregation.
- The “neutral evaluator” role resembles a judge/aggregator that must integrate arguments from multiple agents—relevant if GALILEO uses any adjudication, voting, or deliberation mechanism.

## How is this different from GALILEO?

- This is primarily a *social-science-style empirical study* of conformity in LLM debates, not an optimization method for reliable multi-agent coordination.
- Outcome is “stance adoption” on contentious issues, not task performance / correctness.
- Uses “intelligence” as a proxy for model scale/capability; may not control for prompt/role framing confounds as tightly as an engineering-focused system.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is designed for robustness (e.g., calibrated aggregation, debiasing, diversity constraints), it can explicitly counteract the majority/capability dominance effects this paper documents.
- Engineering contribution (if present in GALILEO): explicit mechanisms and measurable task metrics can yield clearer causal conclusions than open-ended stance outcomes.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses a single judge/aggregator model, this paper suggests it may be disproportionately influenced by a stronger model’s rhetoric or by majority coalition effects.
- If GALILEO assumes “more agents = better,” this paper provides evidence that increasing agent count without balancing capability/diversity can *worsen* conformity/bias.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation: hold total token budget fixed, vary (a) number of agents and (b) capability mix, and measure whether the aggregator’s outputs show conformity-like drift.
- [ ] If using a judge/arbiter, test “single strong agent vs. many weaker agents” conditions to see whether GALILEO’s decision rule is robust to capability dominance.
- [ ] Consider enforcing diversity constraints (capability and viewpoint) in agent sampling, and report it as a mitigation against bias amplification.

## Quotes / details to potentially cite

- Abstract claim (paraphrase-ready): “LLM agents tend to align with numerically dominant groups or more intelligent agents… a single high-intelligence agent can influence a neutral agent more effectively than a group of lower-intelligence agents.”
- Scale/setup: “over 2,500 debates” on “five socially contentious topics,” with a neutral centrist agent evaluating each turn.
