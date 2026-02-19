# Social Contagion and Bank Runs: An Agent-Based Model with LLM Depositors

- Year: 2026
- Venue: arXiv
- Authors: Christopher Ruaño
- URL: https://arxiv.org/abs/2602.15066
- BibTeX key (if we add it): ruano2026socialcontagion
- Tags: agents, social-influence, agent-based-modeling, llm-agents, financial-contagion

## One-sentence takeaway

An agent-based bank-run simulator uses a constrained LLM as each depositor’s decision policy (withdraw/stay + optional post) to study how within-bank connectivity and cross-bank social spillovers can create fast cascade/phase-transition dynamics beyond balance-sheet fundamentals.

## What problem does it solve?

- Canonical bank-run theory (Diamond–Dybvig) explains multiplicity but is weak on *process*: how beliefs synchronize/propagate in real time, especially with digital banking + social media.
- Global-games approaches pick equilibria via signal-noise assumptions, but the informational primitives are hard to justify/measure in modern episodes.
- Need a *measurable* “social correlation / communication network” amplification channel to stress-test alongside fundamentals.

## What is the core method / protocol?

- Build a **process-based ABM** with three main blocks:
  - **Bank mechanics**: cash-first withdrawal processing; when liquidity is insufficient, forced liquidation of long-term assets at a haircut (fire-sale discount); operational throughput limits; endogenous “stress index” / perceived default risk that evolves with withdrawal intensity.
  - **Depositor heterogeneity**: risk tolerance; weight on fundamentals vs. social information; insured vs. uninsured status (sampled by bank-specific uninsured shares).
  - **Communication network**: heavy-tailed within-bank network; multi-bank extension includes depositor overlap + “spillover” connections between banks.
- **Depositor behavior via a constrained LLM policy**:
  - Input: a structured description of the depositor’s information set (numeric fundamentals/stress, recent withdrawal rate, insurance status, neighbor posts).
  - Output: a discrete action (withdraw vs. stay) plus optional short post.
  - Authors emphasize strict action interface + ablations, and a conservative validation exercise against a lab bank-run coordination setting (first-round decisions).
- Parameter/sensitivity approach: large sweeps (“4,900 configurations”) plus full LLM simulations.
- Communication-layer tuning: uses March 2023 Twitter/X activity (SVB topic series and spillovers to other tickers) to motivate heavy-tailed influence concentration and asymmetric cross-bank propagation.

## What are the key metrics?

- Run likelihood / failure risk; time-to-cascade / speed of withdrawal cascades.
- Cross-bank contagion intensity (spillover) and tipping behavior.
- Withdrawal rates by depositor type (notably uninsured vs. insured).
- Scenario-level ordering of bank failures (SVB vs First Republic vs a “safe” regional bank) in a disciplined multi-bank setup.

## What are the main results?

- **Within-bank connectivity increases cascade risk and speed**, holding fundamentals fixed: denser communication can synchronize beliefs/actions faster.
- **Cross-bank contagion shows a phase transition**: failure risk “tips” sharply as spillover rates rise, with a reported tipping region around ~0.10.
- **Channels interact nonlinearly**: depositor overlap + network amplification can be weak individually but strong in combination.
- In an SVB/First Republic/regional scenario anchored to crisis-era facts, the simulator reproduces the **ordering of failures** and predicts **substantially higher withdrawal rates among uninsured depositors**.

## How is this similar to GALILEO?

- Treats **information propagation / social influence** as a first-class causal mechanism rather than an exogenous “sunspot”.
- Uses an **LLM as a policy** inside a simulator with a constrained interface, then evaluates/ablates the LLM component.
- Focuses on **network-mediated cascades** and the importance of modeling *who sees what, when*.

## How is this different from GALILEO?

- Domain is **bank runs / financial stability**, with agents as depositors and actions as withdraw/stay, rather than open-domain task agents.
- Relies on an ABM with scenario-specific parameterization (haircuts, uninsured shares, seizure thresholds) and uses Twitter/X data as a tuning anchor.
- LLM is framed as a “general social agent” prior rather than a learned/optimized policy (no RL training / no fine-tuning), and the action space is very small.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes *general* social-influence evaluation and causal identification, it may be less tied to episode-specific scenario design choices.
- A broader action space / richer task settings could test manipulation/persuasion pathways beyond binary withdrawal decisions.

## Where GALILEO is weaker / needs to improve

- This paper offers a concrete recipe for turning social-media evidence into **network structure assumptions** (heavy-tailed influence; spillover estimation), which GALILEO may need analogues for.
- Provides an explicit **tipping/phase-transition** framing for contagion parameters; GALILEO may want similarly explicit “phase diagrams” for social-influence regimes.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite as an example of **LLM-driven ABM agents** with a constrained action interface and explicit communication networks.
- [ ] Consider adding an experiment/reporting section that emphasizes **phase-transition style plots** (tipping regions) for key contagion / communication parameters.
- [ ] Document a “social correlation as exposure” analogy (parallel to correlated assets) as a framing device for GALILEO’s social-influence risk.

## Quotes / details to potentially cite

- Uses a “**process-based agent-based model** … [making] the information and coordination layer explicit” and implements depositors as a “**constrained large language model** that maps each agent’s information set into a discrete withdraw or stay decision and an optional post.”
- Reports that “**cross-bank contagion exhibits a sharp phase transition** … around spillover rates near **0.10**.”
- Communication network is “**heavy-tailed**” and tuned using Twitter/X activity during March 2023; influence concentration motivates hub/core–periphery diffusion assumptions.
