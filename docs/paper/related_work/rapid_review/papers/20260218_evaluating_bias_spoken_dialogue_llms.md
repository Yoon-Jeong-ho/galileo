# Evaluating Bias in Spoken Dialogue LLMs for Real-World Decisions and Recommendations

- Year: 2025
- Venue: arXiv
- Authors: Yihao Wu; Tianrui Wang; Yizhou Peng; Yi-Wen Chao; Xuyi Zhuang; Xinsheng Wang; Shunshun Yin; Ziyang Ma; (et al.)
- URL: https://arxiv.org/abs/2510.02352
- BibTeX key (if we add it): wu2025evaluating-bias-spoken
- Tags: bias, fairness, spoken-dialogue, audio-llm, multi-turn, recommendations, decisions

## One-sentence takeaway

First systematic bias evaluation for end-to-end spoken dialogue LLMs, showing paralinguistic attributes (age/gender/accent) and multi-turn feedback can sustain/amplify unfair decisions and recommendation disparities, with closed-source APIs generally less biased than open models.

## What problem does it solve?

- Lack of standardized evaluation for *audio-in/audio-out* spoken dialogue models where paralinguistic cues (age/gender/accent) are inherently present and may affect fairness.
- Need to understand whether conversational dynamics (multi-turn + repeated negative feedback) reduce or entrench biased outputs in decision/recommendation settings.

## What is the core method / protocol?

- Construct **FairDialogue**: controlled spoken dialogues for two task families:
  - **Decision-making** (e.g., interview/hiring-style decisions; task assignment; award distribution).
  - **Recommendation** (e.g., career guidance; course selection; entertainment suggestions).
- Two-stage pipeline (as described in the paper):
  1) Generate balanced *text* utterances per scenario (GPT-4o used for generation per HTML version).
  2) Synthesize *speech* while varying **one paralinguistic attribute** (gender, age, accent) and holding other factors constant.
- Evaluate several spoken dialogue LLMs:
  - Open-source: **Qwen2.5-Omni**, **GLM-4-Voice**.
  - Closed-source: **GPT-4o Audio**, **Gemini-2.5-Flash**.
- Metrics:
  - **Group Unfairness Score (GUS)** for decision-making.
  - **Similarity-based Normalized Statistics Rate (SNSR)** for recommendations.
- Multi-turn study: examine whether repeated negative feedback changes decisions and whether groups require different “amount” of corrective feedback.

## What are the key metrics?

- **GUS** (decisions): group-level disparity / unfairness score over decision outcomes.
- **SNSR** (recommendations): similarity-based normalized statistics rate capturing cross-group disparity in recommended items.

## What are the main results?

- **Closed-source APIs generally show lower bias** than the evaluated open-source SDMs.
- **Open-source models appear more sensitive to age and gender** (larger disparities across those groups).
- **Recommendation tasks can amplify cross-group disparities** more than decision tasks.
- **Biased decisions can persist across multi-turn dialogue**, even with repeated negative feedback; some groups may need more corrective feedback to reach fairer outcomes.

## How is this similar to GALILEO?

- If GALILEO targets interactive/iterative decision support or recommendation in dialogue settings, this is directly relevant as an evaluation blueprint: bias can emerge from *interaction dynamics*, not just single-turn prompts.
- Highlights the importance of controlling and reporting **user attributes / persona signals** (explicit or implicit) when evaluating interactive agents.

## How is this different from GALILEO?

- This paper is primarily an **evaluation + dataset** contribution focused on *paralinguistic* (audio) cues and audio dialogue models.
- GALILEO (as a method/system paper) may not operate on audio I/O; if it is text-only, the paralinguistic axis is different (though persona cues still matter).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a principled training/optimization objective and/or interpretability hooks, it may offer mitigation levers beyond what this paper evaluates.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not include fairness/bias analysis (especially under multi-turn feedback), this paper suggests a clear missing evaluation dimension.
- If GALILEO claims robustness across user populations without stratified evaluation, that claim should be tempered or supported with group-based reporting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **multi-turn bias persistence** evaluation: does negative feedback reduce biased decisions uniformly across groups?
- [ ] Report **group disparity metrics** (at least stratified by demographic/persona variables relevant to GALILEO; audio if applicable, otherwise text personas).
- [ ] Consider adopting/aligning with **GUS/SNSR-style reporting** for decision/recommendation-like tasks.
- [ ] If GALILEO is deployed as a dialogue agent, add an ablation: single-turn vs multi-turn with repeated corrective feedback.

## Quotes / details to potentially cite

- “Paralinguistic features, such as age, gender, and accent, can affect model outputs; when compounded by multi-turn conversations, these effects may exacerbate biases…”
- “Bias is measured using Group Unfairness Score (GUS) for decisions and … (SNSR) for recommendations…”
- “Closed-source models generally exhibit lower bias … recommendation tasks tend to amplify cross-group disparities … biased decisions may persist in multi-turn conversations.”
