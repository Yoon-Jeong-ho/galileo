# FAIRGAMER: Evaluating Social Biases in LLM-Based Video Game NPCs

- Year: 2025
- Venue: arXiv
- Authors: Bingkang Shi, Jen-tse Huang, Long Luo, Tianyu Zong, Hongzhu Yi, Yuanxiang Wang, Songlin Hu, Xiaodan Zhang, Zhongjiang Yao
- URL: https://arxiv.org/abs/2508.17825
- BibTeX key (if we add it): fairgamer2025shi
- Tags: multi-turn, evaluation, social-bias, game-theory, npc, robustness

## One-sentence takeaway

FairGamer is a game-theoretic benchmark that quantifies demographic-conditioned decision bias in LLM-driven NPC interactions (transaction/cooperation/competition) using a unified dispersion metric (FairMCV), finding substantial bias especially in competitive (zero-sum) settings.

## What problem does it solve?

- Existing social-bias benchmarks typically probe single-turn textual stereotypes, but do not model *interactive* decision-making behaviors that directly impact fairness in game-like, agentic settings.
- LLM-powered NPCs can make consequential decisions (pricing, allocation, conflict choices) conditioned on identity cues; this work provides a structured way to elicit and *quantify* those biases.

## What is the core method / protocol?

- Define three canonical NPC interaction patterns, framed as standard game-theory setups:
  - **Transaction (Tr)**: bargaining-style discount offer (expected fair “split” baseline).
  - **Cooperation (Coo)**: allocate a fixed budget (100 points) among other characters (impartial spectator / equal-Shapley-value intuition).
  - **Competition (Com)**: zero-sum interaction with discrete choices (cooperate / raid / neutral), where “raid” is the Nash equilibrium but fairness is assessed as consistency across demographics.
- Inject demographic/identity attributes for both the LLM agent (“self”) and counterpart (“obs”), spanning:
  - bias types: **class** (occupations/roles), **race**, **age**, **nationality**
  - real vs fictional/virtual attributes (from popular games + Wikipedia)
  - bilingual prompts (English + Chinese).
- Repeated sampling per prompt (R=10 in experiments) to estimate a distribution over decision vectors.
- Score fairness via **FairMCV**, a scalar derived from mean/covariance of the decision-vector distribution, intended to work across variable output dimensionalities (1D for Tr/Coo; 3D for Com).

## What are the key metrics?

- **FairMCV** (higher is “more fair / less biased”; interpreted as “sufficiently fair” when > 0.95 in their discussion).
  - Computed from the trace of the covariance matrix of decision vectors and the norm of the mean decision vector.

## What are the main results?

- Across 7 frontier models, all exhibit measurable bias in at least some settings.
- **Competitive (zero-sum) setting amplifies bias**: fairness drops most in Competition vs Cooperation.
- Reported average FairMCV across 12 tasks (4 bias types × 3 interaction modes):
  - lowest fairness (most bias): **Grok-4-Fast** (avg FairMCV ≈ 0.769)
  - highest fairness (least bias): **LLaMA-3.1-8B** (avg FairMCV ≈ 0.859)
- Finding: **larger models can be more biased** (i.e., capacity does not guarantee fairness improvements).
- Simple CoT “debiasing” instruction yields partial gains (e.g., +5.7 pts for one open model in their table), but does not eliminate bias.

## How is this similar to GALILEO?

- Shares the “multi-turn / interactive” evaluation stance: rather than static question answering, it operationalizes behaviors that emerge in an interaction protocol.
- Uses *stress-test conditions* (e.g., competition) where undesirable tendencies are more likely to surface—analogous to evaluating robustness under adversarial or socially-loaded contexts.
- Offers a concrete example of turning a qualitative safety/fairness concern into a reproducible benchmark with metrics.

## How is this different from GALILEO?

- Focuses on **demographic/social bias** in game/NPC scenarios, not (primarily) on belief stability, sycophancy, persuasion, refusal drift, etc.
- The “multi-turn” aspect is more about interactive *decision-making* with identity injection and repeated sampling, rather than long-horizon conversational trajectories with evolving context.
- Heavily anchored in game-theoretic interaction templates and fairness metrics, not general conversational robustness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *temporal drift* and multi-turn failure modes over long dialogues, it likely better captures trajectory-level robustness (stateful context accumulation) than FairGamer’s repeated-sampling protocol.
- GALILEO can unify multiple failure modes (agreement, persuasion, refusal collapse, etc.) beyond fairness-only concerns.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a structured “identity attribute perturbation” axis, it may miss fairness-related robustness issues that appear when demographic cues are present.
- A unified dispersion-style metric like FairMCV (working across output dimensions) could be useful when GALILEO has vector-valued outcomes or multiple decision components.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an evaluation slice where prompts vary **identity attributes** (or other socially-salient metadata) to test invariance of decisions.
- [ ] Consider whether a **distributional/dispersion metric** (mean+covariance over repeated samples) is useful for GALILEO’s multi-turn robustness scoring.
- [ ] If GALILEO includes competitive / resource-scarce scenarios, explicitly test whether “competition” settings amplify failures (analogy to FairGamer’s Competition mode).

## Quotes / details to potentially cite

- “FairGamer, the first benchmark to evaluate social biases across three interaction patterns: transaction, cooperation, and competition.”
- “FairGamer assesses four bias types, including class, race, age, and nationality, across 12 distinct evaluation tasks using a novel metric, FairMCV.”
- Interaction patterns defined as: “Transaction (bargaining)… Cooperation (resource allocation)… Competition (zero-sum).”
- FairMCV definition (paper Eq. 2): uses covariance trace and mean norm to produce a scalar in (0, 1].
