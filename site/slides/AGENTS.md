# Talks

The repository's `AGENTS.md` and `site/AGENTS.md` apply here too. This guide holds what is specific to the talks,
and it is kept current: when the workflow or a convention changes, change it here in the same commit.

## What lives where

- `deck.js`, `deck.css`, `sequence.js`: the deck every talk runs on (reveal.js at 1280 × 720, the page's light
  identity from `../css/canvit.css`) and its own elements (`<deck-sequence>`); `deck.css` also holds the utilities
  every talk may use (`.draw` arrows, `.marks` rings, `.step-marker`). Slide conventions are documented at the top of `deck.js`: `data-play`,
  `data-canvit-target`/`data-canvit-t`, `<section data-step>`, `<section data-status>`.
- `package.json`: reveal.js, pinned and installed locally (`npm ci --prefix site/slides`), so a talk runs offline.
- `shoot.py`: screenshots of every slide in Chromium, and the browser errors (below).
- `YYYY-MM-DD-Event/`: one directory per talk: `AGENTS.md` (the talk's audience, framings, scoped claims and
  decisions), `OUTLINE.md` (the talk, slide by slide, designed before the slides), `index.html` (every slide, with
  speaker notes), `talk.css` (rules scoped by slide id), `PLAN.md` (the talk's open work, ranked), `sources/` (the
  material it draws on) and any talk-only modules or assets.

Computing, exporting and plotting are separate steps [Yohaï, 2026-10-01]: an experiment writes its results as
uncolored data (labels, probabilities, scalar maps, metrics, viewpoints) to files; figures and slides read those files
and choose colors, colormaps and layout, so a figure can be redrawn differently without running the model again.
This is the web bundle's rule (`canvit-pytorch/docs/viz.md`, "Data layers are lossless and uncolored").

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

Every fact a talk needed checked in the paper, the code or the data is noted in the talk's `AGENTS.md` ("Checked
facts") with where it lives, the same day [Yohaï, 2026-10-01: "save notes for yourself ... for anything and
everything that you needed to check by reading code or paper"], so the next session finds it without rereading.
A note points to its source (paper section, module and symbol, file path) and states the fact; numbers the paper
generates stay in its macros.

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

- Slide text comes from verified human material, in its words: the paper, the project page (`../index.html`: its
  headings and labels, such as "What's in an active-vision model?", "See it in action"), the README, the authors'
  own writing and speech (the applications and talks quoted in a talk's `AGENTS.md` and `sources/`), and the
  sources the paper cites. Look there for every title before writing one [Yohaï, 2026-10-01, on "Three axes of an
  active vision model": "why use such [bad] titles like this when primary material that i gave you say the MUCH MUCH
  MUCH better 'What's in an active-vision model?'"].
- Titles take the written register: the page's and the paper's headings, the authors' written applications, or the
  plain name of what the slide shows. A spoken turn of phrase (a joke, an aside, a rhetorical question such as "dumb,
  but not too dumb") belongs in the notes, where the speaker says it [Yohaï, 2026-10-01, on that title: "[...] you actually put this as the title"].
- Labels are as short as the thing they name: "Paper", "Code", "Models", not "Read the paper". Install commands use
  uv: `uv add canvit-pytorch` [Yohaï, 2026-10-01].
  Reuse their wordings, turns of phrase and framings verbatim where they fit; write new wording only when nothing
  human fits, and keep it plain [Yohaï, 2026-10-01, on the title "Perception goes beyond the information sampled":
  "I HATE the title of that slide it's ugly ai-written [stuff] ... ALWAYS favor wordings, turns of phrase,
  formulations, framings, etc, that come from verified human material"]. A paraphrase of a human sentence is not
  human wording. Read the paper section a slide covers before writing the slide.
- A title names the thing the slide shows ("Canvas Attention", "Scene-Relative RoPE", "A CanViT rollout") or
  states a claim the paper or a cited source makes ("No policy can make up for a poor observer"). Never a
  tagline, a metaphor or a slogan of your own.
- A title is literally true and matches what the slide shows. History is told as the paper tells it: active
  models "have struggled to match" passive ones; they did not "fall behind".
- A setting detail (the glimpse size, a canvas grid) belongs in the body or the notes, never in a title.
- Never number slides, sections or items in outlines, plans and notes; refer to a slide by its title or its id
  [Yohaï, 2026-10-01: "I [...] HATE YOUR TENDENCY TO NUMBER EVERYTHING, that creates unnecessarily huge diffs and
  wastes of tokens just because you move slides around and can be highly misleading"].
- What a talk shows, and how many slides it takes, follows from what makes it correct, clear, meaningful and
  beautiful [Yohaï, 2026-10-01]; no slides-per-minute rule decides it.
- Layouts use the slide: the visual as large as the 1152 × 628 px content area allows, no large empty regions, the
  most important element the largest [Yohaï, 2026-10-01: "avoid wasting space with your layout, always"].
- The last slide stays up during questions: the links (QR codes, largest) and the funding acknowledgments on one
  slide [Yohaï, 2026-10-01].
- Visual storytelling first and always; less is more [Yohaï, 2026-10-01]. A slide states a fact once: no
  information repeated on the same slide (affiliations as text under the faces and again as logos). Check
  centering and balance on the screenshot: reveal.js sets `display: block` inline on the shown slide, so a
  section's own `display: flex` does not apply; center blocks with auto margins or a wrapper.
- Ask what the slide's takeaway is, then make it the most visible thing on the slide; nothing else gets more
  emphasis than it. A number the audience does not need goes to the notes, never into a large callout [Yohaï,
  2026-10-01, on the Canvas Attention slide showing "2.8 GFLOPs" in large type: "huge visual callout to something
  almost no one in the audience understands or cares about, with wrong emphasis ... WHAT IS THE TAKEAWAY HERE? it's
  the asymmetry"]. A paper figure that carries the point is shown as large as the slide allows, with annotations
  (rings, `.marks`) pointing at the part that matters.
- Quality over quantity [Yohaï, 2026-10-01: "still so much useless, ugly, poorly presented [stuff] in your
  slides"]. The deck holds only slides whose visual carries the point. A slide that would be a list of lines, a
  paragraph or a placeholder box stays out of `index.html` until its visual exists; its plan lives in `OUTLINE.md`
  and `PLAN.md`.
- The neuroscience is the paper's: human vision is active (gaze shifts toward regions of interest, sequential,
  with strategic planning, integration across time in visual working memory, and top-down recurrent
  feedback), each point with its citation.
- For any slide that matters, try several ways to present its point, render each (screenshot, or play the
  animation), compare what works and what does not, then integrate the best parts [Yohaï, 2026-10-01: "in general i
  would recommend trying a bunch of different ways to do something, rendering them / playing with it, then seeing
  what works and what doesn't and integrating together"]. Variants are drafted outside the deck (an untracked page in
  the talk's directory, deleted after the choice) and the choice is recorded in `OUTLINE.md`.
- Demonstrate, then name [Yohaï, 2026-10-01: "such things should always be your first thoughts"]. The first idea for
  any claim is how to show it happening, on a real image, with the real model or the real baseline: for "pasting
  local predictions into scene coordinates does not extrapolate", a photo of a table, two crops, each crop's passive
  segmentation flying to its place in a scene-wide map, and the hole left in the middle. A claim shown only as text
  or a diagram is a placeholder until such a demonstration exists or is ruled out.
- No subtext [Yohaï, 2026-10-01, three times: "remove this useless subtext", "superfluous subtext"]: on-screen text
  is the title, the labels a visual cannot be read without, and the citation of another paper's figure. A caption
  that restates the visual, a takeaway line, a conditions line, and a citation of the authors' own paper go to the
  notes.
- The takeaway is said, not written: speaker notes (`<aside class="notes">`) hold the spoken text, in full
  sentences, and say only what the paper or a cited source supports.
- Glimpse blue, canvas red and policy teal keep their meanings on every slide (`../css/canvit.css`).
- A figure from another paper is cited on the slide (`.cite`: authors, year, venue).
- Animations play while their slide is shown and pause when it is left (`data-play`); a slide that builds
  click by click uses fragments, so the replay button and the speaker view stay in step.
- Interactive figures (the live model, the charts' tooltips) must not advance the deck when clicked: they are
  listed in `INTERACTIVE` in `deck.js`, or carry `data-no-advance`.
