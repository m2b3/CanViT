# Talks

The repository's `AGENTS.md` and `site/AGENTS.md` apply here too. This guide holds what is specific to the talks,
and it is kept current: when the workflow or a convention changes, change it here in the same commit.

## What lives where

- `deck.js`, `deck.css`: the deck every talk runs on (reveal.js at 1280 × 720, the page's light identity from
  `../css/canvit.css`). Slide conventions are documented at the top of `deck.js`: `data-play`,
  `data-canvit-target`/`data-canvit-t`, `<section data-step>`, `<section data-status>`.
- `package.json`: reveal.js, pinned and installed locally (`npm ci --prefix site/slides`), so a talk runs offline.
- `shoot.py`: screenshots of every slide in Chromium, and the browser errors (below).
- `YYYY-MM-DD-Event/`: one directory per talk: `AGENTS.md` (the talk's audience, framings, scoped claims and
  decisions), `OUTLINE.md` (the talk, slide by slide, designed before the slides), `index.html` (every slide, with
  speaker notes), `talk.css` (rules scoped by slide id), `PLAN.md` (the talk's open work, ranked), `sources/` (the
  material it draws on) and any talk-only modules or assets.

A visual that shows CanViT itself (recorded rollouts, features, the live model, the paper's results) is a
`<canvit-*>` component in `../js/canvit/`, usable on the project page as well; a talk never keeps its own copy.
Talk-only visuals (a history timeline, a title slide) stay in the talk's directory. Data come from the same places
as the page's: bundles under `../data/` recorded by `canvit_pytorch.viz`, the paper's exports under
`../assets/paper/`, and numbers in `<span data-macro>` checked by `../check_paper_numbers.py`. A number the paper
does not carry yet (a rebuttal result promised for the camera-ready, a new experiment) has an `XXX` comment beside
it naming its source.

## Workflow

[Yohaï, 2026-10-01] A talk is designed in its `OUTLINE.md` before it becomes slides: for every slide, the
proposition it states or proves, what is on screen, the visual and how it builds, what is said, the keywords, the
sources and the status. The outline is reviewed (independent reviewers: fresh subagents, `codex exec` with model
`gpt-6-astra`, asked open questions about understanding, key messages and what they would do) and revised until
the authors are confident in it; slides are built from it, and an outline change comes before a slide change.

```bash
npm ci --prefix site/slides                     # once: reveal.js
node site/serve.mjs 8765                        # serve site/; open pages reload when a file changes
# http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/   (S: speaker view with notes)
uv run site/slides/shoot.py --url http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/ \
    --out site/.screens/main-2026 > site/.screens/shoot.log 2>&1
python3 site/check_paper_numbers.py
```

`shoot.py` exits nonzero when the page logged an error; read `errors.txt` and look at the images of every slide
you touched, with all fragments shown (default) and before them (`--first-fragment`). Screenshots are review
material and are never committed (`site/.screens/` is ignored).

Every slide carries `data-status`: `ready`, `draft` (content in place, visuals or wording to finish) or
`aspirational` (shows what we want; the data or code behind it does not exist yet). Drafting views show it as a
corner badge; `?present` hides the badges. An aspirational slide describes its missing visual in a
`.placeholder` and has an entry in the talk's `PLAN.md`; status changes land in the same commit as the work.

## Writing slides

[Yohaï, 2026-10-01, on the first draft of the MAIN 2026 talk: invented titles such as "Seeing is something you do",
"the world you build is spatiotopic" and "A glimpse is always 128 × 128 pixels" were rejected outright.]

- Slide text comes from the paper and the sources it cites, in their words. Read the paper section a slide
  covers before writing the slide.
- A title names the thing the slide shows ("Canvas Attention", "Scene-Relative RoPE", "A CanViT rollout") or
  states a claim the paper or a cited source makes ("No policy can make up for a poor observer"). Never a
  tagline, a metaphor or a slogan of your own.
- A title is literally true and matches what the slide shows. History is told as the paper tells it: active
  models "have struggled to match" passive ones; they did not "fall behind".
- A setting detail (the glimpse size, a canvas grid) belongs in the body or the notes, never in a title.
- What a talk shows, and how many slides it takes, follows from what makes it correct, clear, meaningful and
  beautiful [Yohaï, 2026-10-01]; no slides-per-minute rule decides it.
- The neuroscience is the paper's: human vision is active (gaze shifts toward regions of interest, sequential,
  with strategic planning, integration across time in visual working memory, and top-down recurrent
  feedback), each point with its citation.
- The takeaway is said, not written: speaker notes (`<aside class="notes">`) hold the spoken text, in full
  sentences, and say only what the paper or a cited source supports.
- Glimpse blue, canvas red and policy teal keep their meanings on every slide (`../css/canvit.css`).
- A figure from another paper is cited on the slide (`.cite`: authors, year, venue).
- Animations play while their slide is shown and pause when it is left (`data-play`); a slide that builds
  click by click uses fragments, so the replay button and the speaker view stay in step.
- Interactive figures (the live model, the charts' tooltips) must not advance the deck when clicked: they are
  listed in `INTERACTIVE` in `deck.js`, or carry `data-no-advance`.
