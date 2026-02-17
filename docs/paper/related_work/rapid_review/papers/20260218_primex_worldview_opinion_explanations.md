# PrimeX: A Dataset of Worldview, Opinion, and Explanation

- Year: 2025
- Venue: EMNLP 2025 (Main)
- Authors: Rik Koncel-Kedziorski; Brihi Joshi; Tim Paek
- URL: https://arxiv.org/abs/2510.00174
- BibTeX key (if we add it): primex2025
- Tags: opinions, user-modeling, personalization, worldview, explanations, survey

## One-sentence takeaway

PrimeX is a public-opinion dataset augmented with (i) free-text human explanations and (ii) Primal World Beliefs (worldview) per respondent, and shows these extra belief signals measurably improve personalized opinion prediction with LMs.

## What problem does it solve?

- Existing “opinion prediction / persona-adapted LM” datasets typically have limited per-person signal: few topics per person and little higher-level belief context beyond demographics.
- This limits (a) cross-topic generalization for user modeling and (b) analysis of what aspects of a person’s belief system are useful for personalization/alignment.

## What is the core method / protocol?

- Data collection from **858 US residents** (diverse by region/age/education/gender) in a ~30 minute session.
- Each participant answers:
  - **Public opinions**: subsets of questions from **3 Pew American Trends Panel waves** (10 selected questions per wave; preference for personal-opinion questions with higher-entropy response distributions).
  - **Explanations**: for **3 questions per survey** (so 9 total), participants write free-form explanations (“draw on personal history… beliefs, values”, etc.).
  - **Worldview**: **Primal World Beliefs Inventory (PI-18)** measuring top-level (Good) and secondary (Safe/Enticing/Alive) primals.
- Empirical analyses (as described in the paper):
  - Correlate worldview with opinions and with explanation style.
  - Use LMs to predict a user’s held-out opinions, comparing user-representation variants that include (a) only opinions, (b) opinions + explanations, (c) opinions + worldview, (d) opinions + explanations + worldview.
  - Analyze “utility” of a given explanation for improving prediction.

## What are the key metrics?

- Opinion prediction performance on held-out survey questions (exact metric not captured from the arXiv HTML excerpt; likely accuracy / log loss depending on setup).
- Correlation / association analyses between primals, opinions, and stylistic properties of explanations.
- Predictability of primals from opinions/explanations (model performance on primal prediction).

## What are the main results?

- **Explanations help**: adding human-written “why I believe this” text improves a persona-adapted LM’s ability to predict other opinions of the same individual.
- **Worldview helps**: Primal World Beliefs correlate with opinions across topics and influence explanation style; incorporating primals into the user representation further improves prediction.
- **Primals are inferable**: a user’s primals can be predicted to some extent from their opinions and explanations, suggesting a route to compact user representations.

## How is this similar to GALILEO?

- Both care about **stable, high-level user belief structure** that generalizes across tasks/topics, rather than only surface preferences.
- Highlights the value of **structured latent variables** (here: primals/worldview) and **natural language rationales** (human explanations) as inputs to personalization.

## How is this different from GALILEO?

- PrimeX is **a dataset + initial analysis** for personalization/opinion prediction, not a full alignment/control framework.
- Uses **survey opinions** as primary supervision; GALILEO likely targets broader interactive alignment objectives beyond predicting survey answers.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses explicit constraints/guarantees or formal objectives for safe personalization, it can go beyond PrimeX’s empirical “what signals help” framing.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a clean way to incorporate **worldview-level latent factors** or to evaluate **cross-topic stability**, PrimeX provides concrete instruments (PI-18) and an evaluation scaffold.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **worldview latent** (compact factors) to GALILEO’s user model; PI-18-style factors are a plausible template.
- [ ] Add an evaluation slice: **predict held-out opinions across distinct topics**; measure gains from (i) explanations, (ii) worldview, (iii) both.
- [ ] If using rationales, distinguish **human explanations vs model-generated explanations** and quantify which is beneficial under what conditions.

## Quotes / details to potentially cite

- “PrimeX … survey data from 858 US residents with two additional sources of belief information: written explanations … and the Primal World Belief survey for assessing respondent worldview.”
- Motivation framing: worldview as “powerful, compact, and predictive model of the individual’s belief system.”
