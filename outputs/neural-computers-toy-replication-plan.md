# Neural Computers (arXiv:2604.06425) — marimo competition implementation plan

## Summary
The goal is **not** a faithful full-scale replication of the paper. For the alphaXiv × marimo notebook competition, the right target is a **CPU-friendly interactive notebook** that brings one core idea of the paper to life.

The official competition page explicitly says:
- participants should **not** try to fully reproduce the paper,
- the notebook should focus on a **key contribution**,
- submissions are **CPU-only**,
- judges prefer notebooks that provide **intuitive understanding** through **code, UI elements, and text**, with bonus credit for **extensions**.

Given those constraints, the strongest submission direction is:

## Recommended notebook concept
**Neural Computers in Miniature: learning terminal dynamics from I/O traces**

Instead of reproducing the paper’s Wan2.1-scale pixel-video models, build a notebook that:
1. generates or loads small terminal trajectories,
2. turns them into aligned training examples,
3. trains a tiny CPU-friendly model to predict short-horizon terminal evolution,
4. visualizes what it learned,
5. explores one paper-inspired phenomenon such as:
   - the effect of stronger conditioning,
   - why terminal appearance is easier than symbolic reasoning,
   - why clean scripted data helps more than noisy data.

This is much more aligned with the competition than a heavyweight “full replication.”

---

## Strongest evidence

### From the official competition page
The competition asks entrants to:
- implement a result in a **marimo notebook**,
- bring a **core idea** to life,
- emphasize **interactive understanding**,
- optionally add an **extension or variant**,
- submit a **molab link**,
- work under **CPU-only** compute.

### From the paper and released code
- The paper’s public repo releases the **data engine**, not the trained NC models.
- The paper’s central practical claim is that early “runtime primitives” can be learned from **aligned I/O traces** in CLI and GUI settings.
- The most reproducible public piece is the **CLI data pipeline** using asciinema/VHS-style traces and structured scripts.
- The paper itself reports that clean, controlled terminal data is particularly useful.

### Practical implication
A competition entry should emphasize:
- **conceptual clarity**,
- **small, inspectable experiments**,
- **interactive controls**,
- **a compelling extension**,
not giant-scale training.

---

## What to implement

## Recommended scope: CLI-only notebook
This is the best balance of feasibility, faithfulness, and contest value.

### Core notebook story
A notebook user should walk away understanding:
1. how a neural computer prototype can be framed as **predicting interface state from I/O traces**,
2. why **clean scripted terminal data** makes the problem learnable,
3. why **rendering / short-horizon dynamics** are easier than **symbolic reasoning**,
4. how stronger conditioning can improve outputs without implying true reasoning.

### Best toy abstraction
For the competition, use one of these two abstractions:

#### Option A — ASCII-grid neural computer (**recommended**)
Represent the terminal as a character grid rather than raw RGB pixels.

Inputs:
- first terminal state,
- action/script tokens,
- optional prompt/caption.

Outputs:
- next terminal states over a short horizon.

Benefits:
- CPU-trainable,
- easier to visualize and debug,
- more educational,
- easier to compare exact correctness,
- still faithful to the paper’s core idea.

Then render the predicted character grid back into a terminal-like visual for the notebook.

#### Option B — tiny pixel-terminal model
Represent terminal frames as low-resolution images and train a very small CNN/ConvLSTM/Transformer.

Benefits:
- visually closer to the paper.

Costs:
- more compute,
- more brittle,
- harder to finish well before the deadline.

### Recommendation
Use **Option A as the primary notebook**, optionally with a small pixel-rendered view for aesthetics.

---

## Why ASCII-grid is a better contest strategy than pixel video

### Verified constraints
- CPU-only competition.
- Judges reward intuitive understanding, interactivity, and extensions.
- Public repo does not release the paper’s full training code/model.

### Inference
A small text-grid model is more likely than a pixel-video model to:
- run comfortably in molab,
- be understandable to judges,
- be reproducible by others,
- support live parameter exploration,
- leave time for a polished extension.

---

## Notebook MVP

## Title idea
**Neural Computers in Miniature: Learning a Terminal from Traces**

## Sections

### 1. Paper idea in one screen
- Explain the paper’s core idea in plain language.
- Distinguish:
  - conventional computer,
  - agent using a computer,
  - neural computer prototype.
- State the notebook scope clearly: **toy CLI replication of learned runtime primitives**.

### 2. Build the toy dataset
- Generate small terminal sessions from scripts.
- Examples:
  - `pwd`
  - `date`
  - `whoami`
  - `echo $HOME`
  - simple Python REPL arithmetic
- Show:
  - action script,
  - terminal states over time,
  - rendered terminal frames.

### 3. Train a tiny model
Suggested CPU-friendly model options:
- GRU/Transformer over terminal grid tokens,
- small seq2seq model over character-grid states,
- optionally a latent-state model that predicts next grid from previous grid + action.

Notebook controls:
- number of training examples,
- sequence length,
- hidden size,
- conditioning mode.

### 4. Interactive rollout demo
User picks a held-out trajectory and sees:
- ground truth next states,
- predicted next states,
- diff heatmap / changed characters,
- rendered animation.

### 5. Main paper insight experiments
At least two of the following:

#### Experiment A — clean vs noisy data
Show that a model trained on cleaner scripted traces performs better than one trained on noisier or more varied traces.

#### Experiment B — conditioning matters
Compare:
- first-frame only,
- first-frame + script,
- first-frame + richer instruction.

#### Experiment C — appearance vs reasoning
Show that the model can often maintain terminal appearance and local dynamics while still failing arithmetic or exact symbolic tasks.

#### Experiment D — reprompting / answer hints
Inspired by the paper’s arithmetic discussion, show how stronger conditioning can sharply improve outputs, while clarifying that this is not evidence of native reasoning.

### 6. Extension (important for judging)
Pick one extension that adds insight:

#### Best extension candidate
**What actually matters more: data cleanliness or model size?**

Interactive sweep:
- vary training set cleanliness / determinism,
- vary model size,
- compare exact accuracy.

This aligns tightly with a paper result and is feasible on CPU.

Other extension options:
- curriculum learning by command complexity,
- new command families,
- OOD command combinations,
- text-grid vs pixel-grid comparison,
- explicit action traces vs caption-only conditioning.

### 7. Takeaways
Summarize observations versus inferences:
- what the toy model learned,
- what it did not learn,
- how this relates to the paper.

---

## Minimal technical design

## Data representation
Recommended primary representation:
- terminal state as fixed-size character grid, e.g. 24×80 or a reduced toy grid such as 12×40.
- vocabulary includes visible ASCII plus special tokens.
- optional side channel for cursor position.

## Input/target format
At timestep `t`:
- input: first state or previous state(s), plus action/script tokens,
- target: next state.

Short horizon:
- 5–20 steps per episode.

## Model choices
Preferred order:
1. **GRU baseline** over flattened grid tokens + action embeddings.
2. **Tiny Transformer** baseline.
3. Optional latent-state variant with explicit recurrent hidden state `h_t` to mirror the paper’s NC framing.

## Metrics / oracles
The notebook should explicitly define success checks.

### Verified-oracle set
- **Overfit oracle:** can the model memorize 16–32 examples?
- **Held-out exact character accuracy**
- **Held-out exact line accuracy**
- **Action-conditional accuracy after command boundaries**
- **Human visual spot-checks** inside notebook

### Nice-to-have
- rendered GIF or frame-by-frame animation,
- difference overlay / colorized mismatch view,
- confusion breakdown by command type.

---

## What is verified, inferred, and missing

### Verified
- Public repo exposes CLI and GUI trajectory generation, including VHS-style CLI scripts.
- Paper gives enough detail to justify a toy CLI notebook focused on aligned I/O traces.
- Competition is CPU-only and favors intuitive, interactive notebooks over full reproductions.

### Strong inference
- A text-grid model is more likely to score well than a small pixel-video model because it better satisfies notebook usability and CPU constraints.
- A strong extension around **data quality vs model size** is both contest-friendly and faithful to the paper.

### Missing for faithful replication
- full NC model training code,
- released checkpoints,
- exact Wan2.1 training setup,
- full ablation implementations.

These do **not** block the recommended notebook submission.

---

## Recommended execution environment
For this competition objective, my recommendation is:

## **Virtual environment**
Why:
- easiest for marimo authoring,
- best for rapid iteration,
- simpler than Docker for interactive notebook development,
- easy to export to a molab-compatible notebook later.

Docker is still an option if you want strict reproducibility, but it is not my first choice for this contest workflow.

---

## Concrete next steps once environment is confirmed

### Phase 1 — notebook scaffold
- create marimo notebook structure,
- add explanatory text and paper framing,
- add data-generation section.

### Phase 2 — toy dataset
- implement small scripted CLI trajectory generator,
- create train/val/test splits,
- add visualization widgets.

### Phase 3 — baseline model
- implement small GRU/Transformer model,
- add train/eval loop suitable for CPU,
- run overfit oracle.

### Phase 4 — insight experiments
- conditioning ablation,
- clean-vs-noisy ablation,
- arithmetic/reprompting demo.

### Phase 5 — polish for competition
- make notebook self-explanatory,
- add interactive controls,
- improve visual presentation,
- ensure runtime is manageable in CPU-only molab.

---

## Recommended decision
If your goal is the competition, I recommend:

## **Build a CPU-friendly marimo notebook around a CLI text-grid toy NC, not a paper-faithful video model.**

That is the highest-probability way to:
- actually finish by the deadline,
- learn the paper deeply,
- submit something interactive and elegant,
- and still stay genuinely faithful to the paper’s central idea.

---

## Pending decision from user
Before I implement anything, choose the execution environment:
- Local
- Virtual environment
- Docker
- Plan only

My recommendation: **Virtual environment**.

---

## Sources
- Competition page: https://marimo.io/pages/events/notebook-competition
- marimo docs: https://docs.marimo.io/
- Paper abstract/page: https://arxiv.org/abs/2604.06425
- Paper PDF: https://arxiv.org/pdf/2604.06425
- Project essay/blog: https://metauto.ai/neuralcomputer/
- Public repo: https://github.com/metauto-ai/NeuralComputer
