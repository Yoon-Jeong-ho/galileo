# WebLINX: Real-World Website Navigation with Multi-Turn Dialogue

- Year: 2024
- Venue: arXiv (also on OpenReview)
- Authors: Xing Han Lù, Zdeněk Kasner, Siva Reddy
- URL: https://arxiv.org/abs/2402.05930
- BibTeX key (if we add it): lu2024weblinx
- Tags: multi-turn, web-navigation, agents, benchmark, html-pruning, multimodal

## One-sentence takeaway

WebLINX introduces a large-scale, real-website, multi-turn conversational web-navigation benchmark and a “Dense Markup Ranker” (DMR) for pruning HTML, showing finetuned smaller models can outperform strong zero-shot LMMs but still generalize poorly to unseen websites.

## What problem does it solve?

- Defines and operationalizes **conversational web navigation**: an agent must follow user instructions over **multiple dialogue turns** while controlling a browser to complete real tasks on real websites.
- Provides a **large benchmark** to train/evaluate such agents, addressing that prior web agents are often limited in scale, realism, or interaction patterns.
- Tackles the practical bottleneck that LLMs/LMMs cannot ingest full HTML pages each step due to context limits.

## What is the core method / protocol?

- **Dataset / benchmark:** ~100K interactions from ~2300 expert demonstrations, spanning **150+ real-world websites** and diverse interaction patterns.
- **Agent setup:** at each step, the model can condition on:
  - system instruction + user dialogue,
  - action + conversation history,
  - screenshot,
  - HTML page (after pruning).
- **Dense Markup Ranker (DMR):** a retrieval-inspired model that **ranks/selects relevant HTML elements** to form a compact page representation for the action model.
- **Action model:** predicts the next action (e.g., click, type, navigate, or respond); output is parsed into executable structured commands.
- **Evaluation:** compares many action models (text-only and multimodal; small finetuned models through proprietary LMMs).

## What are the key metrics?

- Paper frames this as **replicating expert behavior** / successful web task execution; reported results are organized via their leaderboard.
- Key axes of evaluation emphasized in the paper/webpage:
  - overall performance on in-domain test,
  - **generalization to unseen websites** (notably weak),
  - comparisons across text-only vs multimodal and zero-shot vs finetuned.

## What are the main results?

- **Finetuning matters a lot**: smaller finetuned decoders can beat the best zero-shot large models (including GPT-4V per abstract).
- Even finetuned multimodal models that were pretrained on screenshots can be outperformed by smaller finetuned models (per abstract).
- **Generalization remains the main failure mode**: all finetuned models struggle on **unseen websites**.

## How is this similar to GALILEO?

- Same broad problem family: **LLM-driven agents** operating in complex environments with partial observability and long horizons.
- Shares a central systems concern: **context bottlenecks** and the need for **state compression / retrieval** (DMR-style selection) before reasoning/acting.
- Provides an evaluation setting where robustness/generalization is critical, aligning with GALILEO-style claims around reliability and out-of-distribution behavior.

## How is this different from GALILEO?

- WebLINX is primarily a **benchmark + data collection + agent pipeline for web navigation** (HTML + screenshot + action prediction).
- It is grounded in **browser control** with structured actions; GALILEO may focus on different environments/tasks or different forms of supervision/objectives.
- Their key technical component is **HTML element ranking (DMR)**; GALILEO’s core contribution (as positioned in our paper) may not be DOM-specific.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets broader task families beyond web navigation, it may claim **wider applicability** than a DOM/screenshot-centric pipeline.
- If GALILEO emphasizes principled generalization mechanisms, WebLINX results can be used as evidence that current finetuning-heavy approaches still fail OOD.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks strong demonstrations/benchmarks in interactive settings, WebLINX is a reminder that **large-scale, realistic interactive data** can be a differentiator.
- If GALILEO currently lacks a strong **input pruning / retrieval** component, DMR-like selection is a concrete, task-relevant technique.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding WebLINX as a **related benchmark** (or as an additional eval) when arguing about interactive agents and OOD generalization.
- [ ] If we discuss context limitations, cite WebLINX as an explicit example motivating **HTML/page pruning** prior to action prediction.
- [ ] If feasible, run a small pilot: evaluate a GALILEO agent or ablation on a subset of WebLINX to test **generalization to unseen sites**.
- [ ] In writing: use their headline finding (“finetuned small > zero-shot big, but OOD still hard”) as supporting evidence for the field’s current gap.

## Quotes / details to potentially cite

- “WEBLINX - a large-scale benchmark of **100K interactions across 2300 expert demonstrations** of conversational web navigation.” (project page / abstract)
- Benchmark spans “**over 150 real-world websites**.” (project page / abstract)
- They propose “conversational web navigation, where a digital agent controls a web browser and follows user instructions … in a **multi-turn dialogue** fashion.” (abstract)
- They design a retrieval-inspired model that prunes HTML by “**ranking relevant elements**” (Dense Markup Ranker / DMR). (abstract + project page)
- “All finetuned models struggle to **generalize to unseen websites**.” (abstract)
