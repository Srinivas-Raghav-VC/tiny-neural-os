# paper-finder skill review

## Source
- Repo: https://github.com/bchao1/paper-finder
- Local copy: `third_party/paper-finder/`
- Main file: `third_party/paper-finder/SKILL.md`

## What it is
This is a **skill prompt**, not a Python package.
It is mainly a reusable instruction file that tells an agent how to:
- search for papers using web search + Semantic Scholar
- organize results into topic folders
- keep a memory bank of papers
- build a mind graph
- maintain a `references.bib`

## Practical read
For this project, it is useful as a **workflow template** for literature review and reference organization.
It is **not** a drop-in modeling library for our notebook baselines.
It will not directly help us train the toy CLI models.

## Most useful parts for us
1. A stronger structure for paper tracking if we do a bigger related-work sweep.
2. A reminder to search papers from multiple angles and with synonym families.
3. A clean folder pattern for paper notes and references.

## Less useful right now
- It does not help with Transformer / Mamba implementation.
- It does not give training code for sequence models.
- It overlaps somewhat with the alpha-backed paper tools we already have.

## Recommendation
Keep the local copy as reference material in `third_party/paper-finder/`.
Use it later if we want a cleaner paper-memory workflow for the final report or related-work appendix.
For the immediate next step, the higher-leverage work is still:
- implement a tiny Transformer baseline,
- try a Mamba-style or true `mamba-ssm` baseline on the GPU VM if dependency setup is tolerable,
- compare all baselines on the exact same toy metrics.
