# The generative revolution: AI foundation models in geospatial health—applications, challenges and future research

- Year: 2025
- Venue: *International Journal of Health Geographics* (Editorial / vision paper)
- Authors: Bernd Resch et al.
- URL: https://ij-healthgeographics.biomedcentral.com/articles/10.1186/s12942-025-00391-0
- BibTeX key (if we add it): Resch2025GenerativeRevolutionGeospatialHealth
- Tags: geospatial, foundation-model, multimodal, public-health, surveillance, survey

## One-sentence takeaway

A short vision/survey arguing that **geospatial foundation models** (beyond text-only LLMs and beyond imagery-only remote-sensing FMs) are a key missing piece for next-gen digital/public-health surveillance, and it catalogues applications + challenges + a future research agenda.

## What problem does it solve?

- Frames a gap: most “foundation model” progress is **text-centric** (LLMs) or **vision-centric** (remote sensing / imagery), while *geospatial context* (spatial relations, flows, autocorrelation, heterogeneous covariates) is often neglected in health/public-health modeling.
- Motivates why “geospatial digital health surveillance” needs models that can integrate heterogeneous spatial signals for tasks like outbreak detection, risk mapping, and health-disparity analysis.

## What is the core method / protocol?

- Not a new algorithm/benchmark; it is a **state-of-the-art overview + agenda**.
- Organizes discussion around:
  - types of FMs (language, vision/geospatial-vision, multimodal),
  - geospatial health use-cases,
  - key challenges in making geospatial FMs practical (data, modeling, compute, ethics),
  - opportunities such as conversational interfaces / RAG for geospatial catalogs (their “GeoX-GPT” demonstrator example).

## What are the key metrics?

- None (no primary quantitative evaluation; mostly conceptual + illustrative examples).

## What are the main results?

- Core claim: geospatial health has historically been limited by (i) data availability, (ii) modeling complexity (spatial dependence/autocorrelation, spatiotemporal covariance), (iii) disease-specific non-generalizable models, (iv) over-focus on image-only pipelines.
- Argues FMs can mitigate these by enabling transfer + few/zero-shot adaptation across tasks and modalities, and by supporting new interaction paradigms (LLM/agentic interfaces) for data discovery and workflow assistance.
- Emphasizes **multimodal geospatial foundation models** as a future direction for digital health surveillance and assessment.

## How is this similar to GALILEO?

- Broad thematic overlap only:
  - Both care about **foundation models** and **multimodal integration**.
  - Both implicitly motivate “systems” that integrate heterogeneous signals and support human-facing workflows (they discuss agentic / conversational interfaces for geospatial catalogs).

## How is this different from GALILEO?

- This is **public-health / geospatial-health** positioning work, not a controlled evaluation of LLM behavior under interaction pressure.
- No benchmark, protocol, or metrics comparable to GALILEO’s evaluation apparatus.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s strengths (relative to this paper) are likely:
  - precise operationalization + measurement,
  - controlled experimental protocols,
  - quantitative comparisons/ablations.

## Where GALILEO is weaker / needs to improve

- If GALILEO makes any “foundation models in the wild” motivation claims, this paper is a reminder to explicitly discuss **multimodal, domain-specific context** (here: geospatial) and the practical barriers (data governance, bias, privacy, interpretability).

## Action items for GALILEO (experiments / method / writing)

- [ ] If we include a “why foundation models matter” background section, consider a short paragraph on how *domain context* (e.g., geospatial structure) often gets lost in text-only FM narratives; cite this as an example of domain communities calling for “context-aware FMs”.
- [ ] Optional: mine their challenge list (data limits, spatial dependence, ethics) as a generic checklist for “non-text modalities + context integration” limitations.

## Quotes / details to potentially cite

- Abstract-level positioning (good for background/motivation): foundation models have “mostly focused on understanding and generating text, while geospatial features, interrelations, flows and correlations have been neglected,” motivating research into “Geospatial Foundation Models” for digital health surveillance.
