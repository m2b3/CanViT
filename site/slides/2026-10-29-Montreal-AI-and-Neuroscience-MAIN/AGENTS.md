# MAIN 2026: Toward Active-Vision Foundation Models

The talk at MAIN 2026 (Montreal AI and Neuroscience), 2026-10-29: 25 minutes and 5 of questions, Part I by Sabrina
Du, Part II by Yohaï-Eliel Berreby. How to make a talk is in `../AGENTS.md`; this guide adds what is specific to this
one. The talk's positions and the deck as it stands are in `TALK.md`; open work stays in the agent's private memory.
`CLAUDE.md` is a symlink to this file.

## Before any work

Read the paper (`../../../paper/latex/CanViT_Toward_AVFMs.tex`) in full, appendix included, and `TALK.md`; again after
every context reset. View the paper's main figures (`../../../paper/latex/figures/exported/`: `canvit_overview.pdf`,
`arch_overview.pdf`, `canvas_attention_combined.pdf`, `main_results.pdf`, `resolution_and_mask_size.pdf`).

## Where knowledge lives

| File | Holds |
|---|---|
| `TALK.md` | The authors' positions, the deck as it stands, notes provenance, rehearsal timings. |
| `sources/checked-facts.md` | Facts about CanViT, its paper, its code and the talk's measurements, with where each lives. |
| `sources/*.json` | Data the deck reads, each with its provenance. |
| `experiments/` | What computes the deck's data. |
| `assets/*/PROVENANCE.md` | Where each figure, logo and font comes from. |

- A fact names its source and the date it was checked; an interpretation is marked as one, with whose it is. A
  subagent's reading is an inference until checked.
- The authors' positions are written in clean words, never as pasted quotes; a formulation they gave for the talk is
  kept as given. Nothing in the deck is approved unless the authors discussed it.
- Remove what is stale in the step that makes it stale; keep hard-won facts (a number read in a paper, a
  measurement) even when no slide uses them.
- Unpublished results beyond what the deck shows, private sources, and anything about CanViT 2 beyond broad strokes
  (eccentricity-dependent acuity, dynamic scenes) stay out of the repository.

## Working with the authors

- Quick back and forth: ask less, and more carefully; stay at the level the discussion is at.
- Scope claims to what they cover; no superlatives without the comparison that earns them; think a causal claim
  through before stating it; do not invent objections an expert would not raise.
- Take nothing at face value, including the authors' and agents' earlier statements.
- A review re-derives the argument: what problem each section answers, what each slide must establish and why it
  comes where it does. Moving slides follows from that; it is not the review.
- How CanViT is trained is Yohaï's to present: agents put no training details on the slides that define an
  Active-Vision Foundation Model.

## Commands

With `TALK=2026-10-29-Montreal-AI-and-Neuroscience-MAIN`, the commands of `../AGENTS.md`, and, from the repository
root:

```bash
bash site/slides/$TALK/experiments/build_deck_data.sh [EXPERIMENT ...]   # the deck's data, all or the named ones
uv run site/slides/export_talk.py --talk $TALK --name 2026-10-29_MAIN2026_Toward-AVFMs_Berreby-Du
```

An experiment's step runs from this directory as
`uv run --no-sync --project ../../../canvit-pytorch python -u -m experiments.<experiment>.<step>`; `build_deck_data.sh`
names the inputs each needs. Sweeps write to
`site/.experiments/` (ignored by git).
