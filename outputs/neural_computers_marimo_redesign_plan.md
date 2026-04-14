# Neural Computers Mini — curiosity-first marimo redesign plan

## Objective
Turn the current prototype into a more visually legible, curiosity-driven marimo app that progressively unwraps the idea like a flower:

1. start with an intuitive question,
2. reveal the toy world,
3. let the reader poke the system,
4. separate mechanics from semantics,
5. show where the model succeeds and fails,
6. end with a bigger-picture connection back to the paper.

## Design principle
Do **not** present the notebook as a sequence of static charts.
Present it as an **interactive guided investigation** with progressive disclosure.

## Proposed notebook structure

### Petal 0 — landing / hook
A strong opening hero section with:
- one-sentence claim
- one interactive mini-demo
- one plain-English question:
  - "Can a model learn how a computer behaves just by watching screens change?"

### Petal 1 — the world
Show the toy terminal as a living system:
- command family selector
- variant selector
- step-through animation
- toggles for clean/noisy traces
- highlighted changed cells
- compact explanation of each action type: typing, backspace, enter

### Petal 2 — what is easy vs hard?
Make the core distinction visual:
- **mechanics** panel: typing and cursor-like local updates
- **semantics** panel: Enter-triggered command execution
- side-by-side examples where the model handles typing but fails on execution

### Petal 3 — how the model thinks
Expose the local-update model in a friendly visual way:
- current screen patch
- action/context inputs
- predicted next-cell changes
- error heatmap over the screen
- optional changed-cell mask visualization

### Petal 4 — baselines arena
Let the reader switch between:
- copy baseline
- heuristic baseline
- learned MLP
- future GRU baseline

Need a common evaluation view with:
- same episode
- same step
- side-by-side prediction panels
- compact metric cards

### Petal 5 — experiments garden
Use tabs or accordion sections for the three main studies:
- conditioning
- noise transfer
- unseen paraphrases

Each study should include:
- the question in plain English
- the setup in one paragraph
- the chart
- one honest takeaway
- one skeptical takeaway

### Petal 6 — failure gallery
A dedicated section for the most interesting mistakes:
- wrong output after Enter
- brittle paraphrase failure
- arithmetic mismatch
- drift over multiple steps

This is important because it makes the notebook feel scientific instead of promotional.

### Petal 7 — zoom back out
Reconnect the toy to the paper:
- what this toy captures
- what it leaves out
- why a GRU/Transformer/Mamba-like model is the next logical step
- what would count as stronger evidence

## Marimo-specific features to lean on
Grounded by official docs, useful features include:
- `mo.ui.tabs` for study switching
- `mo.accordion` for progressive disclosure
- `mo.callout` for honest caveats / key takeaways
- `mo.lazy` to defer expensive visual sections
- app-style layout and drag/drop presentation for polished app flow

## Visual language
- dark terminal aesthetic for screens
- consistent highlight colors:
  - blue = model prediction
  - green = correct changed cells
  - orange = Enter / semantics
  - red = errors
- metric cards instead of raw tables where possible
- fewer paragraphs, more visual chunks

## Immediate implementation priorities
1. Rebuild the top of the notebook as a **guided story**, not a paper-summary intro.
2. Add a **mechanics vs semantics** explainer section with visual comparisons.
3. Replace plain dataframe dumps with clearer cards / charts / switchable views.
4. Add a **failure gallery** section.
5. Repackage the fixed studies into tabs/accordion blocks.
6. Leave hooks for a future GRU result section without making the notebook depend on it today.

## Success criteria
The redesigned notebook should make a judge quickly feel:
- "I understand the question."
- "I can play with the system."
- "I can see the difference between shallow imitation and real command understanding."
- "This is more than a static report; it is an explorable idea."

## Canonical next step
Implement the redesign in `notebooks/neural_computers_mini.py`, preserving CPU-only execution and existing result files.
