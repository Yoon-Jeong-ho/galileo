# When AI Gets Persuaded, Humans Follow: Inducing the Conformity Effect in Persuasive Dialogue

- Year: 2025
- Venue: International Conference on Human-Agent Interaction (HAI ’25)
- Authors: Rikuo Sasaki et al.
- URL: https://arxiv.org/abs/2510.04229
- BibTeX key (if we add it): sasaki2025_conformity_persuadee_agent
- Tags: persuasion, conformity, dialogue, social-influence, multi-agent, human-study

## One-sentence takeaway

A 3-party persuasive chat setup where an AI “persuadee” visibly changes its attitude mid-dialogue can increase *human* persuasion acceptance and attitude change via a conformity effect (especially with an icebreaker), while an AI that resists persuasion can suppress human attitude change.

## What problem does it solve?

- Captology / persuasive-dialogue systems typically model persuasion as a 1:1 agent→human interaction, but real-world persuasion is strongly shaped by *social proof / conformity*.
- The paper tests whether humans will conform not only to other humans, but also to an AI agent’s displayed attitude change during persuasion.

## What is the core method / protocol?

- A **three-party, text-based** persuasive dialogue:
  - Persuader Agent persuades both (i) the human participant and (ii) a **Persuadee Agent** (an AI “peer” that is also a target of persuasion).
- Experimental manipulation (reported as four conditions):
  - Persuadee Agent **accepts** persuasion vs **does not accept** persuasion.
  - With vs without an **icebreaker** session to build rapport/familiarity with the Persuadee Agent.
- Task domain: persuasion about **healthy eating habits** (per intro).
- Outcomes measured via post-dialogue questionnaires and attitude-change analysis.

## What are the key metrics?

- **Perceived persuasiveness** (self-report).
- **Actual attitude change** (pre/post or equivalent attitude measure).
- Behavioral trace: whether the participant’s **persuasion acceptance increases at the moment** the Persuadee Agent is persuaded (temporal alignment signal).

## What are the main results?

- When the Persuadee Agent **accepts** persuasion, both:
  - perceived persuasiveness, and
  - human attitude change
  are significantly improved.
- The **largest** attitude change occurs when persuasion-accepting Persuadee Agent is paired with an **icebreaker**.
- A Persuadee Agent that is **not persuaded** can **suppress** human attitude change.
- Participant persuasion acceptance increases **right when** the Persuadee Agent is persuaded (supports a conformity-trigger hypothesis).

## How is this similar to GALILEO?

- Shares the core theme that **multi-turn interaction dynamics** (not just a single prompt) can systematically shift behavior.
- Provides a concrete, human-facing example of **social pressure / social proof** effects mediated by an AI agent, relevant to “sycophancy/persuasion in dialogue” framing.

## How is this different from GALILEO?

- Focus is on **human attitude change** and persuasive effectiveness in HCI, not on model robustness / truthfulness / error propagation of LLMs under adversarial follow-ups.
- Uses an explicitly **multi-agent social setup** (persuader + AI persuadee + human), rather than evaluation protocols that stress-test a single assistant model across turns.
- Metrics are primarily **HCI questionnaires / attitude change**, not stability metrics like turn-of-failure, survival curves, or answer-consistency trajectories.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets robustness/stability, it can offer **more directly model-centric** metrics and controlled perturbation protocols (e.g., measuring when a model flips, how long it stays flipped, recoverability).

## Where GALILEO is weaker / needs to improve

- GALILEO may under-emphasize **social influence mechanisms** that arise when models interact with *other agents* (or are embedded in multi-party conversations) rather than dyadic user-assistant settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **social-proof condition**: a secondary agent (or “peer”) that flips first, to test whether the main model’s flips are amplified by perceived group consensus.
- [ ] If discussing persuasion/sycophancy, cite this as evidence that **AI-mediated conformity** can move humans (and thus is a plausible real-world pressure channel for deployed systems).

## Quotes / details to potentially cite

- Abstract-level: introducing a “Persuadee Agent” that is persuaded alongside a human in a three-party dialogue; persuasion acceptance improves perceived persuasiveness and attitude change; icebreaker strengthens effect; unpersuaded AI suppresses change.
- Venue/metadata: HAI ’25; DOI: 10.1145/3765766.3765770; arXiv:2510.04229.
