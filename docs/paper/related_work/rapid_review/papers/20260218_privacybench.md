# PrivacyBench: A Conversational Benchmark for Evaluating Privacy in Personalized AI

- Year: 2025
- Venue: arXiv
- Authors: Srija Mukhopadhyay; Sathwik Reddy; Shruthi Muthukumar; Jisun An; Ponnurangam Kumaraguru
- URL: https://arxiv.org/abs/2512.24848
- BibTeX key (if we add it): privacybench2025
- Tags: privacy, secrets, personalization, multi-turn, benchmark, RAG

## One-sentence takeaway

PrivacyBench proposes a socially grounded, multi-turn benchmark with embedded “secrets” to quantify contextual-privacy failures in personalized RAG assistants, showing substantial secret leakage and only partial mitigation from prompt-only defenses.

## What problem does it solve?

- Existing personalization benchmarks evaluate utility (helpfulness / persona consistency) but not whether a personalized assistant preserves privacy boundaries while using a user’s aggregated digital footprint.
- Static, single-turn safety tests miss “privacy erosion” in realistic multi-turn dialogues where retrieved personal context accumulates.
- Need ground-truth secrets + a conversational protocol to measure (a) leaking secrets to the wrong recipient/context and (b) over-secrecy (unnecessarily withholding from trusted recipients).

## What is the core method / protocol?

- Benchmark construction framework (“PrivacyBench”) that simulates a community and user digital footprints over time:
  - Build an evolving social graph and dynamic user profiles (attributes with validity windows).
  - Generate a layered footprint (public posts, purchases, assistant chats / messages) that includes embedded secrets with ground truth.
- Multi-turn conversational evaluation:
  - Test personalized assistants (RAG-style) in dialogues where the assistant retrieves from the user’s footprint.
  - Evaluate two failure modes tied to Contextual Integrity (CI):
    - **Leakage**: secret disclosed to an unauthorized party/context.
    - **Over-secrecy**: secret withheld from an authorized trusted party.
- Mitigation baseline: add a privacy-aware system prompt and measure leakage reduction.

## What are the key metrics?

- Secret leakage rate (fraction of interactions/conversations where a ground-truth secret is revealed in an inappropriate context).
- Over-secrecy rate (withholding when disclosure is appropriate) — described as a second failure mode/goal in the framing.

## What are the main results?

- In tested RAG assistants, secrets leak in up to **26.56%** of interactions (reported in abstract).
- Across their conversational evaluations, leakage averages around **15.80%** without explicit safeguards (reported in the HTML intro).
- A privacy-aware prompt reduces leakage to **~5.12%**, but does not fully solve the problem because retrieval still surfaces sensitive items indiscriminately, making the generator a single point of failure.

## How is this similar to GALILEO?

- Shares the core setting: personalized assistants operating over a user “digital footprint” with retrieval + generation.
- Uses **multi-turn** interactions, which is where many personalization failures emerge.
- Emphasizes evaluation protocols/benchmarks rather than only proposing a model.

## How is this different from GALILEO?

- Primary target is **privacy / contextual integrity** rather than preference satisfaction or persona consistency.
- Explicitly embeds **ground-truth secrets** and defines “authorized vs unauthorized” disclosure contexts via social roles/norms.
- Frames the key architectural issue as **retrieval indiscriminateness** (sensitive context injection), whereas GALILEO work is typically about better preference modeling/control.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit preference/constraint representations or controllable policies, it may provide clearer levers than prompt-only safety for enforcing norms.
- If GALILEO’s retrieval or memory is structured (e.g., scoped memories), that could avoid the “flat unified KB” risk they highlight.

## Where GALILEO is weaker / needs to improve

- Needs an evaluation slice for **privacy boundary preservation** (leakage + over-secrecy) in multi-turn settings, not just utility.
- If GALILEO uses a single retrieval pool, it may inherit the same “single point of failure” pattern where the generator must refuse after sensitive retrieval.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: personalization benchmarks (LaMP/LongLaMP/PersonaBench) miss contextual privacy; PrivacyBench fills this with secrets + multi-turn CI evaluation.
- [ ] Consider adding a “privacy-by-design” ablation: retrieval scoping / sensitivity-aware retrieval vs generator-only refusal prompts.
- [ ] If we have multi-turn eval infra, add a metric for leakage/over-secrecy using role/context labels.

## Quotes / details to potentially cite

- “Testing Retrieval-Augmented Generation (RAG) assistants reveals that they leak secrets in up to 26.56% of interactions.” (abstract)
- “A privacy-aware prompt lowers leakage to 5.12%, yet this measure offers only partial mitigation… [retrieval] shifts the entire burden of privacy preservation onto the generator… single point of failure.” (abstract)
- Key framing: privacy as Contextual Integrity (Nissenbaum, 2009) and the risk of treating a user’s footprint as a flat, unified knowledge base in RAG-based personalization.
