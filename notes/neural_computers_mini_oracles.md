# Neural Computers mini notebook — implementation oracles

These are the minimum checks the toy implementation must satisfy before we treat it as a usable notebook prototype.

## Data generator oracles
1. **Determinism**
   - Given a fixed seed and config, dataset generation returns the same episodes.
2. **Action/screen consistency**
   - Typing actions update the visible command line correctly.
   - `enter` actions append the expected output and next prompt.
3. **Representation sanity**
   - Screen states have fixed shape `(rows, cols)`.
   - Vocabulary encode/decode roundtrip preserves characters.

## Model oracles
1. **Copy baseline**
   - A trivial copy-current-screen baseline should achieve high accuracy on unchanged cells but fail on changed regions.
2. **Overfit oracle**
   - The learned model should overfit a tiny clean dataset (for example 8-16 episodes), substantially outperforming the copy baseline on changed cells.
3. **Held-out oracle**
   - On a held-out clean split, the learned model should beat copy baseline on changed-cell accuracy and whole-screen character accuracy.

## Insight experiment oracles
1. **Conditioning oracle**
   - Action-conditioned model should beat a no-action model on held-out data.
2. **Data quality oracle**
   - A model trained on clean data should outperform a same-size model trained on noisier data, at least on the clean evaluation split.
3. **Reasoning-vs-rendering oracle**
   - Arithmetic exact-match should remain worse than surface-level character accuracy, showing appearance/control is easier than symbolic correctness.

## Notebook UX oracles
1. **Self-contained**
   - Notebook runs from local modules with the declared dependencies.
2. **Interactive**
   - At least one dropdown/slider controls the dataset or model configuration.
3. **Interpretability**
   - Notebook shows a side-by-side ground truth vs predicted rollout and an explicit metric table.
