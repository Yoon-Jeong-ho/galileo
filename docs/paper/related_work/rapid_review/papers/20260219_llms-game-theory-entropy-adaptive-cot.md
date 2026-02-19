# LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning

- Year: 2026
- Venue: AAAI 2026 Bridge (Logical and Symbolic Reasoning in Language Models) (arXiv preprint)
- Authors: Tommaso Felice Banfi
- URL: https://arxiv.org/abs/2601.10775
- BibTeX key (if we add it): banfi2026llms
- Tags: game-theory, sequential-decision-making, entropy, uncertainty, adaptive-reasoning, icl, rag

## One-sentence takeaway

Use token-level entropy as a *control signal* to adaptively (i) retrieve more in-context examples and (ii) branch into multi-path reasoning, improving move quality in a sequential decision task (Tic-Tac-Toe).

## What problem does it solve?

- Vanilla prompted LLMs are brittle in sequential decision-making: when confidence is misplaced, a single bad step can irreversibly degrade the trajectory/outcome.
- Fixed-depth CoT / fixed number of retrieved examples wastes compute when the state is easy and under-invests compute when the state is hard.

## What is the core method / protocol?

- Testbed: Tic-Tac-Toe where optimal play is known (minimax), enabling per-state/per-move “ground truth” evaluation.
- Maintain a vector DB of board states paired with minimax-optimal moves.
  - Board encoding: flatten 3x3 board to a 9-d vector.
  - Learn a latent embedding with an autoencoder, plus a contrastive objective that clusters states sharing the same optimal move and separates others.
  - Retrieve Top-k nearest states (cosine similarity) to provide in-context exemplars.
- Entropy-guided *adaptive context retrieval*:
  - Use token-level predictive entropy as an uncertainty estimate.
  - Low uncertainty => retrieve fewer examples (compact context).
  - High uncertainty => retrieve more examples (more guidance).
- Entropy-guided *adaptive CoT*:
  - Low uncertainty => concise single-path reasoning.
  - High uncertainty => expand to multiple reasoning paths (branching exploration), then choose action.
- Practical loop details:
  - If generated move is invalid: re-prompt; if still invalid, fall back to a random valid move.

## What are the key metrics?

- Game outcome over a batch of games (win=+1, tie=0, loss=-1), reported as average score.
- Move optimality (via comparison to minimax-optimal move for the given state).
- Compute proxy: number of LLM queries per game.
- Correlation between token-level entropy and move optimality.

## What are the main results?

- Against a sub-optimal algorithmic opponent (100 games):
  - Baseline LLM average outcome: -11.6%
  - Entropy-guided adaptive reasoning: +9.5%
  - Improvement reported as statistically significant.
- Observed negative association between token-level entropy and move optimality (higher entropy tends to coincide with worse moves), supporting entropy as a useful trigger.
- Claims to maintain relatively low number of LLM queries per game (i.e., adaptivity avoids always-on heavy reasoning).

## How is this similar to GALILEO?

- Same high-level philosophy: *allocate compute / context adaptively* based on uncertainty rather than using a fixed reasoning budget.
- Uses retrieval + reasoning together, and treats retrieval as a controllable knob.
- Evaluates robustness/decision quality over sequential interactions rather than single-shot QA.

## How is this different from GALILEO?

- Domain is a tiny perfect-information game (Tic-Tac-Toe) with exact minimax labels; GALILEO likely targets more complex/realistic tasks where optimal actions are not exactly known.
- Retrieval DB is constructed from latent embeddings of game states and is supervised by minimax-optimal moves; GALILEO may use different supervision/feedback signals.
- Uses token-level entropy as the primary uncertainty measure; GALILEO might use other uncertainty proxies or calibration methods.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO operates in open-world or partially observable settings, its evaluation and claims may transfer better than this “toy-but-controlled” environment.
- If GALILEO avoids requiring a labeled optimal-action DB, it may scale better.

## Where GALILEO is weaker / needs to improve

- Consider adding explicit *uncertainty-triggered branching and retrieval expansion* (if not already present), with clear ablations: fixed-k retrieval vs adaptive-k; single-path vs multi-path.
- Add analyses tying the uncertainty signal to error modes (here, entropy correlates with suboptimal moves).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation grid: {fixed retrieval budget vs adaptive} x {single-path vs multi-path} and report compute vs quality trade-offs.
- [ ] Include a “uncertainty signal validity” plot: uncertainty vs correctness/optimality (and calibration) to justify adaptive triggers.
- [ ] If we have retrieval: experiment with adaptive-k retrieval conditioned on uncertainty.

## Quotes / details to potentially cite

- “The model dynamically adjusts both the number of retrieved examples and reasoning paths according to token-level uncertainty: concise reasoning with minimal context is used when uncertainty is low, whereas higher uncertainty triggers expanded multi-path CoT exploration.” (Abstract)
- Reported outcome improvement: average game outcome from -11.6% to +9.5% over 100 games (win=+1, tie=0, loss=-1). (Abstract)
- Method detail: ~20% of all possible Tic-Tac-Toe board states stored in the vector DB with minimax-optimal moves; cosine-sim Top-k retrieval. (Method Section 4.2)
