# InfMem: Learning System-2 Memory Control for Long-Context Agent

- Year: 2026
- Venue: arXiv (ICML submission)
- Authors: Mingze Li; Peng Lu; Xiao-Wen Chang; Lifeng Shang; Jinpeng Li; Fei Mi; Prasanna Parthasarathi; Yufei Cui
- URL: https://arxiv.org/abs/2602.02704
- BibTeX key (if we add it): infmem2026
- Tags: agents, memory, long-context, retrieval, control, early-stopping, RL

## One-sentence takeaway

InfMem turns bounded-memory long-document QA into an explicit **System-2 control loop** (PreThink–Retrieve–Write + early stop) and trains the controller with **SFT warmup + verifier/outcome RL**, yielding large accuracy gains up to **1M-token** contexts while reducing inference time via adaptive stopping.

## What problem does it solve?

- Streaming / bounded-memory agents (e.g., MemAgent-style) process ultra-long documents with constant memory, but **passive per-chunk memory overwrites** can drop low-salience “bridging” evidence needed for **multi-hop reasoning** across far-apart segments.
- Pure long-context scaling or vanilla RAG helps capacity/recall, but does not guarantee **evidence consolidation** into a compact working state or **when-to-stop** efficiency.

## What is the core method / protocol?

- **Setting:** single-pass processing of a long document chunk stream {c_t} with a bounded overwrite memory m_t (token budget M). Separately build a fine-grained retrieval index {p_j} (e.g., paragraphs) for global in-document access.
- **Control loop per step t:**
  - **PreThink(q, m_{t-1})** emits a structured control record (a_t, u_t, k_t):
    - a_t ∈ {STOP, RETRIEVE} (memory sufficiency check)
    - u_t: a single retrieval query (if RETRIEVE)
    - k_t: how many units to retrieve (if RETRIEVE)
  - **Retrieve(u_t, k_t; {p_j}) → r_t**: targeted global retrieval within the same document (non-monotonic access to past/future segments).
  - **Write(q, m_{t-1}, c_t, r_t; M) → m_t**: “evidence-aware joint compression” that integrates current chunk with retrieved evidence into the bounded memory.
  - **Early stopping:** if PreThink outputs STOP, terminate and answer from current memory.
- **Training recipe:**
  - **SFT warmup** by distilling protocol-valid trajectories from a stronger teacher using inference-consistent prompting.
  - **Verifier-based RL (GRPO-style)** aligning retrieval/write/stop decisions with end-task correctness + protocol soundness (well-formed calls; memory budget adherence) + an **early-stop shaping reward** (stop soon after memory becomes sufficient).

## What are the key metrics?

- QA accuracy (reported across long-context QA benchmarks) as a function of context length (32k → 1M tokens).
- Efficiency: inference time / latency reduction via **adaptive early stopping**.
- (Training) protocol-soundness verifiers: valid tool-call formatting + memory-budget compliance.

## What are the main results?

- On ultra-long QA benchmarks (32k–1M tokens), InfMem reports consistent gains vs MemAgent across multiple Qwen backbones.
- Reported average absolute accuracy improvements vs MemAgent: **+10.17 / +11.84 / +8.23** points on **Qwen3-1.7B / Qwen3-4B / Qwen2.5-7B**, respectively (per abstract).
- Reported efficiency: **~3.9× faster** on average (up to **5.1×**) via early stopping (per abstract).

## How is this similar to GALILEO?

- Both emphasize **explicit control** over what evidence is kept/used under constraints rather than relying on passive summarization.
- “Monitor–seek–update–stop” maps cleanly onto many agent/controller designs: a **stateful controller** chooses actions based on an intermediate state.
- Reinforces the narrative that the main bottleneck in long-horizon settings is often **control / evidence management**, not raw context capacity.

## How is this different from GALILEO?

- InfMem is specifically a **long-document QA** framework: bounded memory + in-document retrieval + early stopping, evaluated on QA benchmarks (including synthetic long-context scaling and LongBench QA).
- The learning focus is on **controller alignment** (retrieve/write/stop) with an SFT→RL pipeline; the system assumes an internal in-document retrieval index over the same document.
- If GALILEO is centered on different modalities/tasks (or different failure modes), InfMem is best treated as a **methodological neighbor** (control loop + training), not a direct task match.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contributions are primarily about domain-specific modeling, datasets, or evaluation protocols outside QA, those may be more directly applicable than InfMem’s QA-specific pipeline.
- If GALILEO avoids reliance on a separate retrieval index, it may be architecturally simpler.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses passive memory updates or fixed schedules, InfMem suggests adding:
  - an explicit **sufficiency monitor** (when to stop / when to seek more evidence)
  - **learned retrieval-sizing** (k_t) and query synthesis
  - a training signal that ties intermediate control decisions to end-task success.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **STOP/CONTINUE** controller for any iterative evidence aggregation pipeline, and report compute/latency trade-offs (accuracy vs steps).
- [ ] If applicable, test **non-monotonic access** to previously seen information (in-document retrieval) vs purely streaming updates.
- [ ] If we have multi-stage actions, consider a **two-stage post-training** story: SFT for protocol adherence + RL for end-task alignment, with explicit protocol verifiers.

## Quotes / details to potentially cite

- Abstract framing: streaming agents “often fails to preserve low-salience bridging evidence required for multi-hop reasoning.”
- Protocol: “PreThink–Retrieve–Write” with early stopping; PreThink outputs {STOP, RETRIEVE} plus a query and a retrieval size; Write performs “evidence-aware joint compression” into bounded memory.
- Reported headline numbers (abstract): +10.17 / +11.84 / +8.23 accuracy points across Qwen backbones; ~3.9× speedup (up to 5.1×) via adaptive early stopping.
