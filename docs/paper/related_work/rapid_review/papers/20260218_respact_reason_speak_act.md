# ReSpAct: Harmonizing Reasoning, Speaking, and Acting Towards Building Large Language Model-Based Conversational AI Agents

- Year: 2024
- Venue: arXiv
- Authors: Vardhan Dongre, Xiaocheng Yang, Emre Can Acikgoz, Suvodip Dey, Gokhan Tur, Dilek Hakkani-Tür
- URL: https://arxiv.org/abs/2411.00927
- BibTeX key (if we add it): respact2024dongre
- Tags: agents, ReAct, conversational, human-in-the-loop, task-oriented dialogue, ALFWorld, WebShop

## One-sentence takeaway

ReSpAct extends ReAct by adding an explicit “Speak” action channel so an LLM agent can actively collaborate with a user (clarify, request guidance, confirm assumptions, provide status) and measurably improves success across interactive benchmarks.

## What problem does it solve?

- Standard LLM agent frameworks (e.g., ReAct) reason and act, but treat interaction with a human as limited (mostly clarification) or external to the policy; this leads to brittle behavior when instructions are underspecified, preferences matter, or failures require recovery.
- Need a simple, schema-free way to incorporate user feedback into the agent loop in fully conversational task solving.

## What is the core method / protocol?

- Redefines part of the “language action” space to include *dialogue actions* (Speak) in addition to internal Thoughts.
- Agent interleaves three modes:
  - **Reason/Think**: internal language steps.
  - **Speak**: natural-language utterance to user (questions, confirmations, status updates, preference elicitation, etc.).
  - **Act**: environment/tool actions.
- When the agent speaks, the user response is appended to observations/context, shaping subsequent reasoning/actions.
- Implementation is prompt-based (few-shot exemplars) with frozen LLMs; evaluated with either a user simulator (for scale) or humans (for WebShop in one setting).

## What are the key metrics?

- **ALFWorld**: task success rate (%) across task types.
- **MultiWOZ**: Inform (%) and Success (%) for task-oriented dialogue.
- **WebShop**: score and success rate (SR %).

## What are the main results?

- From abstract: vs ReAct, absolute success-rate gains of **+6% (ALFWorld)** and **+4% (WebShop)**; and **+5.5% Inform / +3% Success (MultiWOZ)**.
- In the HTML text (v2) they report ALFWorld with GPT-4o: **ReSpAct best-of-6 87.3% vs ReAct 80.6%**; average **85.3% vs 79.4%** (Table 1).
- MultiWOZ (Table 2 excerpt shown): with GPT-4o-mini, **Inform 72.2 vs 66.7**, **Success 51.8 vs 48.8**; ReSpAct uses more turns (~6.5 vs 5.1).
- WebShop (Table 3 excerpt shown): GPT-4o-mini: ReSpAct with human interaction can be dramatically better than simulator (SR 50% human vs 12% user-sim), highlighting simulator limitations.

## How is this similar to GALILEO?

- Shares the “agent loop” framing: interleave language reasoning with actions, and treat the agent as operating in an environment with observations/actions.
- Emphasizes robustness under ambiguity and failure recovery via iterative interaction.

## How is this different from GALILEO?

- Primary contribution is *interaction modeling*: an explicit Speak action and incorporating user responses into the observation stream; not primarily about new planners, world models, or verification.
- Largely prompting/few-shot based; improvements come from better conversational behavior rather than algorithmic guarantees.
- Evaluation focuses on interactive settings (TOD + ALFWorld/WebShop with a user simulator), rather than offline planning-only settings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has more formalized state tracking, tool grounding, or verification, that could be positioned as complementary: ReSpAct improves *how* to ask/confirm, while GALILEO improves *what* to do / correctness.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently assumes fully specified tasks (or limited user dialogue), this paper suggests explicit “preference/assumption confirmation” and “status update” behaviors can yield measurable gains.
- User simulation quality matters: ReSpAct shows large gap between simulator and human results on WebShop, so GALILEO should be careful about over-relying on simulators.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “Speak”/“Ask user” action type to the agent interface and ablate: no-speak vs speak-enabled.
- [ ] Create a small taxonomy of conversational moves beyond clarifications (status update, preference elicitation, fallback/alternative suggestion, failure explanation) and test which matter.
- [ ] In related-work, cite ReSpAct as a direct extension of ReAct for conversational collaboration in interactive environments.

## Quotes / details to potentially cite

- “ReSpAct employs active, free-flowing dialogues … without any explicit dialogue schema.” (Abstract)
- “By alternating between environment actions, language thoughts, and dialogue actions, the agent interleaves task-solving reasoning with targeted human interaction.” (Section 3 framing)
- Reported improvements: “absolute success rate improvements of 6% and 4% in ALFWorld and WebShop … 5.5% gain in Inform and 3% gain in Success in MultiWOZ.” (Abstract)
