# SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents

- Year: 2026
- Venue: arXiv
- Authors: Yujiong Shen, Yajie Yang, Zhiheng Xi, Binze Hu, Huayu Sha, Jiazheng Zhang, Qiyuan Peng, Junlin Shang, Jixuan Huang, Yutao Fan, Jingqi Tong, Shihan Dou, Ming Zhang, Lei Bai, Zhenfei Yin, Tao Gui, Xingjun Ma, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang
- URL: https://arxiv.org/abs/2602.12984
- BibTeX key (if we add it): sciagentgym2026shen
- Tags: agents, tool-use, scientific, benchmark, multi-step, long-horizon

## One-sentence takeaway

SciAgentGym + SciAgentBench provide a large-scale benchmark for long-horizon *scientific* tool-use, showing sharp performance degradation with interaction length and proposing dependency-graph-based synthetic trajectories (SciForge) to improve tool orchestration.

## What problem does it solve?

- Existing agent benchmarks under-test *domain-specific scientific workflows* that require orchestrating many specialized tools across multiple steps.
- The gap: measuring robustness of multi-step tool execution (state tracking, tool dependencies, compounding errors) in natural science contexts.

## What is the core method / protocol?

- **SciAgentGym**: interactive environment with a large catalog of domain-specific tools (reported: 1,780 tools) spanning four natural science disciplines, with an execution infrastructure to actually run tool calls.
- **SciAgentBench**: “tiered” evaluation that increases difficulty from elementary tool actions to long-horizon workflows.
- **SciForge**: data synthesis approach that models the tool action space as a **dependency graph**, generating “logic-aware” training trajectories; fine-tune a smaller model (SciAgent-8B) on these trajectories.

## What are the key metrics?

- Task success rate across tiers / interaction horizons (short vs long-horizon).
- Comparative performance across model families and sizes; cross-domain transfer (qualitative claim) after fine-tuning.

## What are the main results?

- Strong models exhibit a **large drop** in success when interaction horizons extend (example figure in abstract: 60.6% → 30.9%).
- Fine-tuning on SciForge trajectories yields a smaller agent model (SciAgent-8B) that reportedly **outperforms** a much larger baseline (Qwen3-VL-235B-Instruct) on their scientific tool-use setting.
- Authors claim **positive cross-domain transfer** of tool-use capabilities.

## How is this similar to GALILEO?

- Both emphasize **multi-step / long-horizon** evaluation where failures compound over turns/steps.
- Offers a concrete benchmark framing for “when horizon increases, success collapses”, which parallels GALILEO-style concerns about instability under extended interaction.

## How is this different from GALILEO?

- Focus is **scientific tool-use orchestration**, not belief drift / persuasion / social pressure dynamics (unless GALILEO includes tool-use).
- Primary failure mode discussed is multi-step tool workflow execution rather than conversational instability metrics (e.g., turn-of-failure or recovery-to-truth).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *conversational drift / susceptibility / recovery*, it is more directly aligned with safety/robustness concerns than domain-specific scientific tooling.
- GALILEO can potentially provide clearer causal controls (evidence vs pressure) that tool-use benchmarks do not separate.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a tool-execution environment, SciAgentGym highlights a direction for **realistic long-horizon execution** with many tools and hard dependencies.
- If GALILEO does not vary interaction horizon systematically, SciAgentBench’s tiered setup is a good pattern to emulate.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small “tool dependency” ablation or auxiliary benchmark: evaluate performance as a function of step count and dependency depth.
- [ ] In related work, cite this as evidence that **horizon length is a primary driver of failure**, even for strong models, motivating GALILEO’s long-horizon stress tests.

## Quotes / details to potentially cite

- “SciAgentGym … featuring **1,780 domain-specific tools** across four natural science disciplines …” (abstract)
- “success rates drop sharply from **60.6% to 30.9%** as interaction horizons extend … primarily due to failures in multi-step workflow execution.” (abstract)
- “SciForge … models the tool action space as a **dependency graph** to generate logic-aware training trajectories.” (abstract)
