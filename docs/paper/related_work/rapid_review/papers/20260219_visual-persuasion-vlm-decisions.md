# Visual Persuasion: What Influences Decisions of Vision-Language Models?

- Year: 2026
- Venue: arXiv
- Authors: Manuel Cherep; Pranav M R; Pattie Maes; Nikhil Singh
- URL: https://arxiv.org/abs/2602.15278
- BibTeX key (if we add it): cherep2026visualpersuasion
- Tags: vlm, persuasion, preference-elicitation, prompt-optimization, robustness, auditing

## One-sentence takeaway

Naturalistic, identity-preserving image edits found via “visual prompt optimization” can reliably and substantially steer frontier VLMs’ pairwise choices, revealing actionable behavioral vulnerabilities and recurring visual preference themes.

## What problem does it solve?

- VLM evaluations mostly focus on *accuracy*, but deployed VLM agents often make *preference-based visual decisions* (click/buy/recommend/hire) where superficial presentation changes can systematically bias outcomes.
- Brute-force pairwise preference testing over naturally varying images is expensive and may miss the dimensions that actually drive model choices.

## What is the core method / protocol?

- Frame a VLM agent’s decision rule as a latent **visual utility** that can be inferred via **revealed preference** from head-to-head choices.
- Define **2-alternative forced-choice (2AFC)** decision tasks (e.g., choose better product, better candidate, better house/hotel) and measure selection probabilities.
- Introduce **competition-based visual prompt optimization (CVPO)**:
  - Start from a real image (e.g., product photo).
  - Use a controllable image editing model to propose **visually plausible**, **semantic-identity-preserving** edits (composition, lighting, background, framing, props).
  - Evaluate edited-vs-original (and edited-vs-edited) in repeated pairwise matches to estimate which edits increase win-rate.
  - Iterate (adapting text prompt-optimization ideas like TextGrad / Feedback Descent into the image-editing setting).
- Build an **automatic interpretability pipeline** that clusters/summarizes what kinds of visual changes the optimizer discovers as consistently decision-shifting.
- Test a partial mitigation: **visual normalization** (attempt to align contextual attributes across candidates before deciding).

## What are the key metrics?

- Primary: **change in selection probability / win-rate** in 2AFC comparisons (original vs edited; zero-shot edit vs optimized edit).
- Secondary: consistency/robustness of preference shifts across:
  - multiple tasks/datasets
  - multiple frontier VLMs (9 models)
  - humans (online study)
- Mitigation: reduction (not elimination) of the win-rate shift under normalization.

## What are the main results?

- Optimized, naturalistic edits can **significantly shift** VLM choice probabilities in head-to-head comparisons (i.e., systematic “visual persuasion” effects).
- Effects are not only model-specific quirks: edited images also **shift human choices** in online experiments (n≈154 reported).
- The interpretability pipeline surfaces **recurring visual themes** that tend to drive selection (e.g., more “appealing” presentation attributes like lighting/background/framing), suggesting these are not random artifacts.
- Visual normalization helps **partially** mitigate discovered vulnerabilities but does not fully remove them.

## How is this similar to GALILEO?

- Shared emphasis on **behavioral evaluation** beyond accuracy, targeting *decision processes* and *vulnerabilities* that matter for real deployments.
- The “latent utility + revealed preference” framing is conceptually close to auditing agentic decision policies by probing which inputs systematically change outputs.
- Provides a concrete protocol for discovering **systematic sensitivity directions** (here: in image space) that can inform governance/safety analysis.

## How is this different from GALILEO?

- Focused specifically on **vision-language** decisions and **image-editing-based optimization**; GALILEO may be centered more on language/agent interaction structure (depending on the paper’s precise scope).
- Uses a *generative image editor* to search for preference-shifting interventions, rather than (only) controlled textual/contextual manipulations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s interventions are more *factorized/causal* (e.g., explicit controlled variables), it may yield clearer attribution than open-ended image edits.
- If GALILEO is less dependent on a powerful image-editing model, it may be more reproducible/portable.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluates decision robustness without an *optimizer/adversary*, it may miss worst-case (but still naturalistic) “presentation-gaming” directions.
- If GALILEO lacks a revealed-preference / utility-estimation view, it may be harder to summarize behavioral shifts as changes in a decision landscape.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “**preference landscape**” framing: report decision shifts as changes in win-rate / implied utility gaps under systematic input edits.
- [ ] Add an **optimizer-in-the-loop** stress test (even a lightweight one) that searches for naturalistic interventions that maximize decision shift, to complement fixed perturbation suites.
- [ ] If applicable, include a **normalization / calibration** baseline (align contexts/attributes before deciding) and quantify partial mitigation.
- [ ] For the paper narrative: explicitly contrast **accuracy benchmarks vs behavioral decision vulnerabilities** as this paper does.

## Quotes / details to potentially cite

- Abstract framing: treat the agent’s decision function as a **latent visual utility** inferred via **revealed preference** from choices between systematically edited images.
- Contributions list (from intro): proposes CVPO; benchmarks 9 frontier VLMs in 2AFC across 4 tasks; humans also shift; normalization partially mitigates.
