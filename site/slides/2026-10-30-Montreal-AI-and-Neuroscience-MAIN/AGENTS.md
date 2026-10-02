# The MAIN 2026 talk

"Toward Active-Vision Foundation Models", MAIN 2026 (Montreal AI and Neuroscience), 2026-10-30. `../AGENTS.md` (how
talks are written and checked) applies.

- `OUTLINE.md`: the talk slide by slide, designed before the slides.
- `PLAN.md`: open work.
- `sources/`: dated research notes, each claim with its source; `sources/checked-facts.md` holds the facts the talk
  relies on (the paper by section, the talk's measurements, code pointers) and the claims to state with their scope.
- `experiments/`: the experiments that compute the deck's data, a package per experiment, each step run from this
  directory as `uv run --no-sync --project ../../../canvit-pytorch python -u -m experiments.<experiment>.<step>`
  (`training_cost` in `../../../paper/exporter`'s environment). Sweeps write to `../../.experiments/` (ignored by
  git); the exports the deck loads go to `../../data/talk/` (deployed, never committed).
  `experiments/build_deck_data.sh` reruns them all, or the named ones.
