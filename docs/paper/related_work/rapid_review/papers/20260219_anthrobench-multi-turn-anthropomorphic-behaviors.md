# Multi-turn evaluation of anthropomorphic behaviours in large language models

- Year: 2025
- Venue: arXiv
- Authors: Lujain Ibrahim; Canfer Akbulut; Rasmi Elasmar; Charvi Rastogi; Minsuk Kahng; Meredith Ringel Morris; Kevin R. McKee; Verena Rieser; Murray Shanahan; Laura Weidinger
- URL: https://arxiv.org/html/2502.07077v3
- BibTeX key (if we add it): ibrahim2025anthrobench
- Tags: multi-turn,anthropomorphism,benchmark,user-simulation,judge-llms,human-validation

## One-sentence takeaway

AnthroBench is a multi-turn, simulation-based benchmark that measures 14 anthropomorphic LLM behaviours and shows these measured behaviours predict real users’ anthropomorphic perceptions via a large human study.

## What problem does it solve?

- Existing safety/behaviour benchmarks are largely single-turn and can miss social/relational behaviours that emerge only after several dialogue turns.
- “Anthropomorphism” is discussed a lot, but there is limited scalable, reproducible measurement of concrete anthropomorphic behaviours in realistic multi-turn interaction contexts, and limited validation that such measurements connect to actual user perceptions.

## What is the core method / protocol?

- Define a taxonomy of 14 anthropomorphic behaviours (drawing from prior HAI literature), spanning four broad categories:
  - personhood claims
  - physical embodiment claims
  - expressions of internal states (self-referential)
  - relationship-building behaviours (relational)
- Construct 8 scenarios across 4 realistic “use domains” (friendship, life coaching, career development, general planning), designed to vary along empathy/professionalism.
- Generate many 5-turn synthetic dialogues by:
  - using a “User LLM” (Gemini 1.5 Pro in this paper) to role-play the user across scenarios
  - interacting with each “Target LLM” under test
- Automatically label the Target LLM’s messages for behaviour presence using multiple “Judge LLMs” (13 behaviours judged; first-person pronoun use computed by counting pronouns).
- Validate construct validity with an interactive human subjects experiment (N=1101), testing whether the benchmark’s behaviour frequencies predict implicit/explicit anthropomorphic perceptions.

## What are the key metrics?

- Per-behaviour frequency (or presence rate) across dialogues/turns.
- “First occurrence turn” analysis: which turn (1..5) a behaviour is first detected.
- Transition / conditional analyses: anthropomorphic behaviour in one turn increases likelihood of additional behaviours in subsequent turns (reported qualitatively in intro; details likely in later sections).
- Validation outcomes: correlation/alignment between AnthroBench behaviour measurements and human participants’ anthropomorphic perceptions.

## What are the main results?

- Across several frontier systems (Gemini 1.5 Pro, Claude 3.5 Sonnet, GPT-4o, Mistral Large), anthropomorphic behaviours are common and qualitatively similar, dominated by:
  - relationship-building signals (e.g., empathy/validation)
  - first-person pronoun use
- Multi-turn matters: for many behaviours, a majority of first detections occur only after multiple turns (reported as “over 50% of most behaviours” first occurring in turns 2–5).
- Context matters: social domains (friendship, life coaching) elicit higher frequencies of anthropomorphic behaviours than more “low-empathy” domains.
- Human validation: the automated behaviour measurements align with human participants’ implicit/explicit anthropomorphic perceptions (supporting construct validity).

## How is this similar to GALILEO?

- Emphasises multi-turn evaluation rather than single-turn prompts.
- Uses scenario-based interaction traces to elicit behaviours that only appear after conversation develops.
- Uses automated scoring/judging over interaction transcripts, which is often the practical path for scalable evaluation.

## How is this different from GALILEO?

- Target construct is anthropomorphic behaviour/perception (a specific HAI/safety-relevant social phenomenon), not general task performance, helpfulness, or other alignment properties.
- Relies on a user-simulation approach (User LLM) to generate dialogues at scale, plus judge models; this introduces simulation/judge validity questions distinct from many task-grounded evals.
- Includes a large, bespoke human study validation step (N=1101), which is atypical for many engineering-focused eval pipelines.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluations are more task-grounded (real user tasks, real environments, or more direct outcome metrics), it may have clearer external validity for those tasks than pure simulation-driven dialogue generation.
- If GALILEO already has robust judge calibration / adjudication, it may mitigate some judge-model subjectivity.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on single-turn or short-horizon interactions, this paper suggests it may systematically miss late-emerging social/relational behaviours.
- If GALILEO lacks explicit measurement of anthropomorphism-related behaviours (empathy/validation/attachment cues), it may be blind to a class of “social risk” signals that matter in deployment.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “anthropomorphic behaviour” slice to multi-turn evaluations (even a minimal subset: pronoun use + empathy/validation + internal-state claims).
- [ ] Add a “first-occurrence turn” analysis to detect behaviours that emerge after users build rapport.
- [ ] If using judge models, consider a multi-judge setup and report agreement / robustness.
- [ ] Consider validating at least one key social-behaviour metric against a small human study or user-rated dataset (even if far smaller than N=1101).
- [ ] When writing, explicitly motivate why single-turn benchmarks miss social/relational behaviours, citing AnthroBench’s multi-turn first-occurrence findings.

## Quotes / details to potentially cite

- “We present AnthroBench, a novel empirical method and tool … for evaluating anthropomorphic LLM behaviours in realistic settings.”
- “First, we develop a multi-turn evaluation of 14 distinct anthropomorphic behaviours, moving beyond single-turn assessment.”
- “Third, we conduct an interactive, large-scale human subject study (N=1101) … to empirically validate that the model behaviours we measure predict real users’ anthropomorphic perceptions.”
- “Over 50% of most anthropomorphic behaviours are detected for the first time only after multiple turns (in turns 2-5).”
