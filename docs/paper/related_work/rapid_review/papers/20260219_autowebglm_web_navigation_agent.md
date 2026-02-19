# AutoWebGLM: Bootstrap and Reinforce a Large Language Model-based Web Navigating Agent

- Year: 2024
- Venue: arXiv
- Authors: Hanyu Lai; Xiao Liu; Iat Long Iong; Shuntian Yao; Yuxuan Chen; Pengbo Shen; Hao Yu; Hanchen Zhang; Xiaohan Zhang; Yuxiao Dong; Jie Tang
- URL: https://arxiv.org/html/2404.03648v1
- BibTeX key (if we add it): autowebglm2024
- Tags: agents, web, navigation, html-simplification, curriculum-learning, dpo, rft

## One-sentence takeaway

AutoWebGLM shows that a compact (6B) LM can become a strong web-navigation agent via (i) an explicit browser action API + simplified-HTML observation, (ii) hybrid human/LM trajectory collection, and (iii) bootstrapping with preference-style RL (DPO) and domain-specific rejection-sampling finetuning.

## What problem does it solve?

- Real-world web navigation remains hard for LLM agents because:
  - action spaces are diverse and inconsistent across sites,
  - raw HTML is extremely long (often 30k+ tokens) and noisy,
  - decision-making is open-domain and error-prone (agents loop / fail to self-correct).

## What is the core method / protocol?

- Agent framework with an explicit **text-based observation** and **function-call action space**.
- Observation includes: task description, **simplified HTML**, current location/scroll info, and previous actions.
- **HTML simplification algorithm**: compresses DOM to preserve key structure/content while removing verbosity (details in appendix in the paper).
- Action space as calls like `click(id)`, `type_string(id, text, enter)`, `scroll_page(direction)`, `jump_to(url, newtab)`, `finish(answer)`.
- Data pipeline (hybrid human + model):
  - Web recognition + single-step operation data constructed with rules + GPT-generated variants.
  - Complex multi-step tasks collected by humans using a browser plugin; step intents/justifications filled using GPT-4 with a global-trace prompting approach.
  - Merge with existing datasets (paper states combining with Mind2Web and MiniWoB++ training data).
- Training in 3 stages:
  1) **Curriculum SFT** from simple -> complex trajectories (learn to read/operate, then plan/reason).
  2) **Self-sampling + DPO**: sample multiple rollouts per task, form positive/negative pairs, apply DPO with an added SFT term for stability.
  3) **RFT (rejection sampling finetuning)** in sandboxed environments (MiniWoB++ / WebArena-style) by sampling many trajectories and keeping successful ones.

## What are the key metrics?

- Step Success Rate (SSR) on:
  - AutoWebBench (bilingual; Chinese/English; in-domain/out-of-domain)
  - Mind2Web
  - MiniWoB++
  - WebArena

## What are the main results?

- AutoWebBench SSR (reported): AutoWebGLM (6B) ~65% on English/Chinese splits and ~59-62% cross-domain/cross-task, substantially above GPT-4 in their setting.
- Mind2Web SSR avg (reported): AutoWebGLM 59.5 (higher than GPT-4 ~30.9 in their table; comparable to larger specialist baselines).
- MiniWoB++ and WebArena: with task-specific finetuning, AutoWebGLM reports strong MiniWoB++ (89.3) and improved WebArena (18.2) versus general baselines.
- Error analysis categories (paper): hallucination (~44%), poor graphical recognition (~28%), task-context misinterpretation (~20%), pop-up interruptions (~8%).

## How is this similar to GALILEO?

- Same general theme: **LLM as an agent** interacting with an external environment via a structured interface.
- Emphasizes that **representation choice** (their simplified HTML) and **action abstraction** (function calls) matter as much as the base model.
- Uses a **bootstrapping loop** where the agent learns from its own rollouts (self-sampling, preference learning, rejection sampling), which parallels the kind of self-improvement / iterative training loops we consider.

## How is this different from GALILEO?

- Domain is specifically **web browsing** with DOM/screenshot grounding and an explicit browser executor; GALILEO may target different environments/tasks.
- Heavy emphasis on **data engineering** for web trajectories + bilingual benchmark creation.
- Their RL is framed mainly as **DPO over sampled trajectories** plus environment-success filtering (RFT), rather than (necessarily) a unified end-to-end world model or planner (depending on GALILEO’s design).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a more general environment interface or cleaner separation between perception/planning/action, it may generalize beyond web-UI settings with less bespoke preprocessing.
- If GALILEO avoids brittle DOM-id style action selection, it may be more robust to webpage drift.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit **state summarization** step analogous to HTML simplification, it may struggle with long-context, noisy observations in UI-like environments.
- If GALILEO does not include a strong **self-correction signal** (preference learning / contrastive negatives), this paper suggests it is important to avoid loops and hallucinated actions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: “compact LM + structured observation/action + bootstrapped preference learning can reach strong web-agent performance” citing AutoWebGLM.
- [ ] Consider an ablation-inspired checklist for our own agent: (a) observation compression, (b) action API design, (c) curriculum (easy->hard), (d) preference learning on self-sampled failures, (e) domain-specialist RFT.
- [ ] If we have any UI/web-like setting, test whether a “simplified state” representation yields large gains vs raw text dumps.

## Quotes / details to potentially cite

- “HTML text exceeding model processing capacity… Token length of content-rich webpages can usually reach 30k and over.” (Intro)
- Contributions summary: HTML simplification + hybrid data construction (~10k traces) + curriculum + RL (DPO) + RFT + bilingual benchmark.
- Action API examples: click/hover/select/type_string/scroll_page/jump_to/switch_tab/finish.
