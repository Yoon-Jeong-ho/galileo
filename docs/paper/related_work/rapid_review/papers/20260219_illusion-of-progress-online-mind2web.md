# An Illusion of Progress? Assessing the Current State of Web Agents

- Year: 2025
- Venue: arXiv
- Authors: Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, Yu Su
- URL: https://arxiv.org/abs/2504.01382
- BibTeX key (if we add it): xue2025illusion
- Tags: web-agents, benchmark, online-eval, llm-as-judge, trajectory-eval

## One-sentence takeaway

A careful online benchmark (Online-Mind2Web, 300 tasks/136 sites) and improved trajectory-based judge (WebJudge) suggest web-agent capability has been overstated by prior benchmarks like WebVoyager.

## What problem does it solve?

- Existing web-agent benchmarks can substantially inflate reported success rates due to limited website/task diversity, shortcut solutions (e.g., Google-searchable tasks), and unreliable automatic evaluation (low agreement with humans).
- Need: (i) an *online* (live web) benchmark that better matches real user usage, and (ii) a scalable automatic evaluator that correlates well with human success judgments.

## What is the core method / protocol?

- **Online-Mind2Web benchmark**:
  - 300 tasks spanning 136 popular websites.
  - Mix of (a) validated tasks adapted from Mind2Web / Mind2Web-Live and (b) newly created tasks.
  - Filters out tasks that have become invalid / ambiguous / CAPTCHA-blocked.
  - Difficulty stratified by human reference trajectory length (easy/medium/hard).
  - Evaluation setting discourages Google Search “shortcuts” by starting agents from a provided URL and prompting them not to use Google Search.

- **WebJudge (LLM-as-a-judge for trajectories)**:
  - Inputs: task, action history, and screenshots along the trajectory.
  - Addresses two failure modes of prior judging:
    1) only-final-screenshot misses intermediate evidence,
    2) using all screenshots causes token overload.
  - Three-stage pipeline:
    1) infer key task requirements (“key points”),
    2) score/select key screenshots (keep relevant intermediate frames),
    3) judge success using key points + key screenshots + actions.
  - Also describes a cheaper learned variant (**WebJudge-7B**) for key-screenshot selection to reduce calls.

## What are the key metrics?

- **Agent success rate (SR)** on Online-Mind2Web (human evaluation as reference).
- **Agreement rate (AR)** between automatic evaluator and human judgments.
- “Gap” between evaluator-predicted SR and human SR.
- Efficiency proxy: steps taken relative to human reference length (reported as an analysis tool).

## What are the main results?

- Under careful human evaluation on Online-Mind2Web, most agents succeed on only ~30% of tasks.
- Best-performing frontier agents in their comparison are **Operator (~61% SR)** and **Claude Computer Use 3.7 (~56% SR)**, while others are clustered near ~30%.
- **WebJudge** reaches about **~85% agreement** with humans and small SR gap (reported ~3–4%), substantially better than prior automatic evaluation approaches.
- Breakdown by difficulty: strong drop-off from easy to medium and hard; frontier agents can do very well on easy tasks but still struggle with long-horizon / complex ones.
- Error analysis highlights common failure modes:
  - filter/sorting mistakes,
  - missing critical finalization steps (submit/apply),
  - navigation errors,
  - misunderstandings,
  - and sensitivity to numeric/temporal constraints.

## How is this similar to GALILEO?

- Shared emphasis on **realistic evaluation** and avoiding benchmark artifacts that overestimate capability.
- Uses **trajectory-level signals** (sequence of observations/actions) rather than only final responses, aligning with agent evaluation needs.
- Provides taxonomy of web-agent failure modes that could map to GALILEO-style robustness / long-horizon reliability analysis.

## How is this different from GALILEO?

- Focus is specifically on **web navigation agents** and their evaluation infrastructure (benchmark + automatic judge), not (necessarily) GALILEO’s target domain.
- WebJudge is tailored to **visual GUI trajectories** (screenshots + action logs); GALILEO may rely on different observability or task definitions.
- Emphasis is on **measuring current capability and correcting inflated claims**, less on proposing a new agent architecture.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has strict, artifact-resistant evaluation (e.g., adversarial / longitudinal checks, non-shortcut tasks), this paper is supporting evidence that such rigor matters.
- If GALILEO can instrument environments beyond screenshots (DOM/state logs, tool outcomes), it may support more faithful automatic evaluation than purely visual judging.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations rely on “final answer” only, or on a limited site/task set, this paper suggests results may be inflated.
- If GALILEO lacks scalable human-aligned automatic evaluation, WebJudge-style design patterns (key requirements + key evidence selection) may be a missing piece.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: *recent work argues WebVoyager may overestimate web-agent capability; Online-Mind2Web shows a large performance drop under more realistic online tasks*.
- [ ] Borrow evaluator design: for any long trajectories, implement **evidence selection** (identify key constraints + select key states) before judging.
- [ ] In robustness analysis, include failure categories similar to theirs (filters/sorting, incomplete steps, numeric/temporal constraints).
- [ ] Consider a “shortcut” baseline (search-only / retrieval-only) to quantify benchmark exploitable shortcuts.

## Quotes / details to potentially cite

- Online-Mind2Web: “300 diverse and realistic tasks spanning 136 websites.”
- WebJudge: automatic evaluation with “around 85% agreement with human judgment.”
- Their central claim: prior benchmarks/evaluators can paint an “over-optimistic” picture; on their benchmark, many agents are around ~30% SR and even strong agents are near ~60%.
