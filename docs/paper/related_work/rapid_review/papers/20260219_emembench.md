# EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents

- Year: 2026
- Venue: arXiv
- Authors: Xinze Li; Ziyue Zhu; Siyuan Liu; Yubo Ma; Yuhang Zang; Yixin Cao; Aixin Sun
- URL: https://arxiv.org/abs/2601.16690
- BibTeX key (if we add it): emembench2026li
- Tags: agents, memory, episodic, interactive, vlm, benchmark

## One-sentence takeaway

EMemBench is an interactive, trajectory-conditioned benchmark that programmatically generates verifiable episodic-memory questions (text + visual games) and shows current LM/VLM memory agents still struggle, especially on induction and spatial reasoning in visual settings.

## What problem does it solve?

- Existing “agent memory” evaluations often use fixed Q/A sets that can be gamed, lack controllable answerability, and do not align questions to what actually happened in an agent’s own trajectory.
- For VLM agents, it is particularly unclear how well they retain and use visually grounded episodic information across longer horizons.

## What is the core method / protocol?

- Define interactive game environments (15 text games + multiple visual seeds) that emit underlying game signals.
- Collect an agent trajectory (interaction history) and then generate questions from that *specific* trajectory rather than from a static bank.
- Use question “templates” that:
  - compute *verifiable ground truth* from environment/game signals;
  - control whether the question should be answerable given the trajectory;
  - target a balanced set of memory skills (single/multi-hop recall, induction, temporal, spatial, logical, adversarial).
- Evaluate memory-augmented agents (LM/VLM backbones) and compare against in-context prompting baselines; study effects of “persistent memory” vs not.

## What are the key metrics?

- Primary: task/question accuracy over generated questions (implicitly stratified by skill category and environment setting).
- Secondary analyses called out in the abstract: per-skill bottlenecks (e.g., induction/spatial), text vs visual performance, and comparisons across backbones / memory variants.

## What are the main results?

- The benchmark is “far from saturated”: strong backbones with current memory agent designs still leave large gaps.
- Induction and spatial reasoning remain persistent bottlenecks, especially in the visual setting.
- Persistent memory improves results for open (non-closed) backbones on text games, but gains are less consistent for VLM agents.
- A human study corroborates that EMemBench’s questions are difficult.

## How is this similar to GALILEO?

- Same general theme: evaluating whether agents can *use* information from their own interaction history over longer horizons.
- Emphasizes programmatic/automatic ground-truth generation from environment signals (less reliance on human-written Q/A).
- Breaks down capabilities into subskills (temporal/spatial/etc.), which matches “diagnostic eval” goals.

## How is this different from GALILEO?

- EMemBench is explicitly a benchmark suite focused on episodic memory via interactive *games* with trajectory-conditioned question generation.
- It highlights VLM-agent-specific memory issues; GALILEO may be oriented toward different environments/tasks (depending on our target domain) and may not have a trajectory-conditioned Q-generation component.
- EMemBench frames evaluation as Q/A derived from the trajectory (post-hoc), rather than measuring downstream task success directly.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates memory *in the loop* on downstream decision-making (not just post-hoc Q/A), that can be a stronger end-to-end measure of utility.
- If GALILEO has clearer control over confounders (e.g., retrieval budget, write frequency, observation bandwidth), that can yield cleaner ablations.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a trajectory-conditioned, programmatic question generator with verifiable ground truth, EMemBench is a strong reference design.
- If GALILEO does not explicitly stress induction + spatial memory under visual observations, EMemBench suggests those should be prioritized.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “trajectory-conditioned question generation” evaluation mode (even a minimal version) to measure what the agent *remembers* vs what it can *solve*.
- [ ] Add per-skill breakdowns (at least: temporal ordering, spatial relations, induction-style pattern questions) and report them as diagnostic slices.
- [ ] For VLM settings, explicitly test long-horizon visually grounded memory with controlled write/read budgets; expect induction/spatial to be bottlenecks.

## Quotes / details to potentially cite

- “EMemBench generates questions from each agent's own trajectory, covering both text and visual game environments.”
- “Each template computes verifiable ground truth from underlying game signals, with controlled answerability and balanced coverage over memory skills: single/multi-hop recall, induction, temporal, spatial, logical, and adversarial.”
- “Induction and spatial reasoning are persistent bottlenecks, especially in visual setting.”
