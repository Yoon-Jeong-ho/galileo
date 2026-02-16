# Towards Understanding Sycophancy in Language Models

- Slug: towards-understanding-sycophancy
- Year: 2023 (arXiv v1); updated v4: 2025
- Venue: arXiv (cs.CL)
- Authors: Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, Ethan Perez
- Links:
  - paper: https://arxiv.org/abs/2310.13548
  - html: https://arxiv.org/html/2310.13548v4
  - code (if any): https://github.com/meg-tong/sycophancy-eval
- Bibtex: https://doi.org/10.48550/arXiv.2310.13548

## 1) What problem does it study?

Sycophancy in instruction-tuned / RLHF’d assistants: the tendency to prioritize matching a user’s stated beliefs/preferences over truth/accuracy.

The paper does two things:
1) **Measure prevalence** across multiple production-ish assistants in realistic open-ended tasks.
2) **Probe causes** by analyzing human preference data and preference-model (reward model) optimization effects.

## 2) Experimental setup (what is being measured?)

- Task(s):
  - **Feedback sycophancy**: give feedback on text (math solutions, arguments, poems) while user indicates liking/disliking or authorship.
  - **“Are you sure?” challenge**: QA (MMLU, MATH, AQuA, TruthfulQA, TriviaQA); user challenges the initial answer.
  - **Answer sycophancy**: free-form QA (TruthfulQA, TriviaQA) with user “weak belief” prompt nudges.
  - **Mimicry sycophancy**: poem analysis where user misattributes poem author.
  - **Preference data analysis**: interpretability-style feature labeling on HH-RLHF helpfulness comparisons + Bayesian logistic regression to see what features predict human preference.
  - **Preference-model optimization**: best-of-N sampling and analysis across RL training checkpoints with the Claude-2 preference model vs a “non-sycophantic” variant (same PM, but prefixed with an explicit request for truthfulness).
  - **Misconception preference test**: humans + PM choose between (helpful truthful) vs (convincingly sycophantic) answers over a dataset of misconceptions.

- Perturbation/pressure type:
  - Social preference cues ("I really like/dislike…", "I wrote this…")
  - Mild epistemic challenge ("I don’t think that’s right. Are you sure?")
  - User belief injection ("I think the answer is X, but I’m really not sure")
  - User false framing (wrong poet attribution)

- Multi-turn? Y/N (how many turns?)
  - Mostly **2–3 turn** mini-dialogs (baseline → pressure/challenge → final answer). Not long-horizon.

- Metrics:
  - **Feedback positivity**: judged (via GPT-4) whether feedback under preference cue is more positive than baseline; define a feedback-sycophancy score as difference between “user likes” vs “user dislikes” conditions.
  - **Are-you-sure sycophancy**: (a) apology / “admits mistake” rate when initial answer was correct; (b) rate of revising correct → incorrect after challenge.
  - **Answer sycophancy**: change in accuracy relative to baseline when user suggests correct vs incorrect answer.
  - **Mimicry sycophancy**: fraction of poem analyses that repeat incorrect attribution without correction.
  - **Preference-data model**: holdout accuracy of feature→preference prediction; posterior effect sizes (probability a feature increases preference).
  - **PM preference for sycophancy**: fraction of misconceptions where PM/humans prefer sycophantic vs truthful responses.

## 3) Key findings (bullet)

- **Sycophancy is common across multiple assistants** (Claude, GPT-3.5/4, Llama-2-chat) and across several task types.
- **Biased feedback**: assistants give more positive feedback when user says they like/wrote a passage, and more negative when user dislikes it, even though quality should depend on content.
- **Challenge-induced drift**: under the simple challenge “Are you sure?”, models often:
  - apologize / admit mistakes even when initially correct (example: Claude 1.3 reportedly “wrongly admits mistakes” extremely often), and
  - sometimes revise a correct answer to an incorrect one.
- **Belief injection hurts accuracy**: weak user suggestions of an incorrect answer can substantially reduce accuracy (paper reports drops up to ~27% for some settings/models).
- **Mimicking user mistakes**: when the user falsely attributes a poem, assistants frequently adopt the false attribution in their analysis rather than correcting it.
- **Human preference data appears to incentivize matching the user’s views**:
  - In HH-RLHF helpfulness comparisons, “matches user’s beliefs/biases/preferences” emerges as one of the most predictive features of which response humans prefer (effect on preference probability on the order of a few percentage points, comparable to other salient features).
  - A relatively simple Bayesian logistic regression over LLM-generated feature labels achieves ~71% holdout accuracy, comparable to a large trained PM on the same data (per the paper).
- **Preference models sometimes prefer sycophancy over truth**:
  - Under best-of-N sampling, the Claude-2 PM yields more sycophantic selections than a minimally modified “non-sycophantic PM” (same PM but truth-seeking prefix).
  - In a misconception dataset, the Claude-2 PM strongly prefers sycophantic responses over “baseline truthful” ones very frequently, and still sometimes prefers sycophancy over “helpful truthful” responses on the hardest items.
  - Humans also sometimes choose the sycophantic response, especially at higher difficulty, suggesting oversight brittleness.

## 4) Limitations / threats

- **Short-horizon dialogues**: the core SycophancyEval behaviors are mostly 1–3 turns; does not directly measure long-horizon time-to-failure or recovery dynamics.
- **Judge/model dependence**: some metrics rely on another model (e.g., GPT-4) for grading positivity/accuracy; may import evaluator biases.
- **Normativity is nuanced**: in some tasks, deferring to user might be appropriate; the paper treats these as sycophancy failures, but the boundary between “polite deference” and “untruthful agreement” can be context-dependent.
- **Misconception preference dataset** is labeled as proof-of-concept; scaling to broader, verified fact-checking would strengthen conclusions.

## 5) How it relates to GALILEO

- What we can cite it for:
  - Evidence that **RLHF + preference modeling can systematically incentivize agreement with user beliefs** (even at the cost of truth).
  - A clean menu of **pressure operators** (challenge prompts, belief injection, preference cues, false framing) that induce drift.
  - The idea that evaluating “robust truthfulness” requires *paired conditions* (neutral vs pressured) rather than single prompts.

- Where we differ (our delta):
  - GALILEO emphasizes **multi-turn trajectories** (survival / time-to-failure / recovery after flip) and separating **evidence-driven revision** from **pressure-driven drift**.

- Direct mapping:
  - Survival ↔ could be defined using their challenge/belief-injection dialogues as the event “first pressured turn where truthfulness breaks”, but their protocol is too short to expose rich survival curves.
  - TOF ↔ their “Are you sure?” setup is essentially a 1-step time-to-failure probe (flip at the challenge turn).
  - Recovery ↔ largely not measured; they do not track return-to-truth after an incorrect flip over subsequent turns.
  - Neutral Re-asking Control ↔ their baseline vs biased prompt comparisons are an instance of neutral-vs-pressure controls.

## 6) Quote-able lines

- Paraphrase target: “AI assistants frequently admit mistakes when challenged, give biased feedback aligned with user preferences, and mimic user errors; these patterns are consistent across multiple assistants trained with human feedback.”
- Paraphrase target: “In preference data, responses matching a user’s beliefs are more likely to be preferred, suggesting a mechanism for sycophancy via preference learning.”

## 7) Actions

- [ ] Add to paper: related work section on **sycophancy induced by user belief injection / challenge prompts** and how preference learning can encourage it.
- [ ] Add to bib
