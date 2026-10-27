# Talks

The repository's `AGENTS.md` and `site/AGENTS.md` apply here too. This guide holds what is specific to the talks,
and it is kept current: when the workflow or a convention changes, change it here in the same commit.

The overarching principle [Yohaï, 2026-10-01]: take nothing at face value; read and understand material fully before
presenting it.

## What lives where

- `deck.js`, `deck.css`: the deck every talk runs on (reveal.js at 1280 × 720, the page's light identity from
  `../css/canvit.css`); `deck.css` also holds the utilities
  every talk may use (`.draw` arrows, `.marks` rings, `.step-marker`). Slide conventions are documented at the top of `deck.js`: `data-play`,
  `data-canvit-target`/`data-canvit-t`, `data-shows`, `<section data-status>`.
- `package.json`: reveal.js, pinned and installed locally (`npm ci --prefix site/slides`), so a talk runs offline.
- `shoot.py`: screenshots of every slide in Chromium, the browser errors and layout problems (below).
- `stress.py`, `check_published.py`: a talk skipped through at speed; its files checked against the published site
  (below).
- `YYYY-MM-DD-Event/`: one directory per talk: `AGENTS.md` (how to work in the talk's directory), `OUTLINE.md` (the talk, slide by slide, designed before the slides), `index.html` (every slide, with
  speaker notes), `talk.css` (rules scoped by slide id), `PLAN.md` (the talk's open work, ranked), `sources/` (the
  material it draws on), `experiments/` (the experiments that compute the talk's data under `../data/talk/`,
  rebuilt by its `build_deck_data.sh`; `deploy.sh` ships the data files the deck names, never the experiments) and any
  talk-only modules or assets.

Computing, exporting and plotting are separate steps [Yohaï, 2026-10-01]: an experiment writes its results as
uncolored data (labels, probabilities, scalar maps, metrics, viewpoints) to files; figures and slides read those files
and choose colors, colormaps and layout, so a figure can be redrawn differently without running the model again.
This is the web bundle's rule (`canvit-pytorch/docs/viz.md`, "Data layers are lossless and uncolored").

A visual that shows CanViT itself (recorded rollouts, features, the live model, the paper's results) is a
`<canvit-*>` component in `../js/canvit/`, usable on the project page as well; a talk never keeps its own copy.
Talk-only visuals (a history timeline, a title slide) stay in the talk's directory. Data come from the same places
as the page's: bundles under `../data/` recorded by `canvit_pytorch.viz`, the paper's exports under
`../assets/paper/`, and numbers in `<span data-macro>` checked by `../check_paper_numbers.py`. A number the paper
does not carry yet (a new experiment) has an `XXX` comment beside it naming its source.

## Workflow

[Yohaï, 2026-10-01] A talk is designed in its `OUTLINE.md` before it becomes slides: for every slide, the
proposition it states or proves, what is on screen, the visual and how it builds, what is said, the keywords, the
sources and the status. Each entry separates what the slide must show, say or carry across (without it the story
breaks or a claim goes unsupported) from what would be interesting or good to show; both get made, and the first
is never cut for the second. The outline is reviewed (independent reviewers: fresh subagents, `codex exec` with model
`gpt-6-astra`, whose sandbox reaches the network only with `-c sandbox_workspace_write.network_access=true`, asked
open questions about understanding, key messages and what they would do) and revised until the authors are confident
in it; slides are built from it, and an outline change comes before a slide change.

```bash
npm ci --prefix site/slides                     # once: reveal.js
node site/serve.mjs 8765                        # serve site/; open pages reload when a file changes
# http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/   (S: speaker view with notes)
uv run site/slides/shoot.py --url http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/ \
    --out site/.screens/main-2026 > site/.screens/shoot.log 2>&1
uv run --no-project python site/check_paper_numbers.py
```

Slide preparation keeps nuance and ambiguity open rather than collapsing them early, gathers primary material, and
makes quick back-and-forth on views into the data easy [Yohaï, 2026-10-01].

A talk is developed on its own branch, which takes in `main` regularly [Yohaï, 2026-10-01].

Every request of the authors that is not yet done goes into the talk's `PLAN.md` ("Open requests") the moment it
arrives, stated plainly, before any work on it, and leaves only when it is done; requests survive the end of a
session only there [Yohaï, 2026-10-01].

Every fact a talk needed checked in the paper, the code or the data is noted in the talk's `sources/checked-facts.md`
with where it lives, the same day [Yohaï, 2026-10-01], so the next session finds it without rereading.
A note points to its source (paper section, module and symbol, file path) and states the fact; numbers the paper
generates stay in its macros.

`shoot.py` exits nonzero when the page logged an error; read `errors.txt`, `layout.txt` (titles that wrap, content in
the footer band, a word alone on the last line of a text block, text blocks that overlap) and the images of every slide you touched, with all fragments shown (default) and before them
(`--first-fragment`), or click by click (`--steps`).
Screenshots are review material and are never committed (`site/.screens/` is ignored).

`stress.py` skips through a talk quickly, forward, back and at random, and checks that each slide lands in the state
a settled visit gives it, with nothing playing off its slide [Yohaï, 2026-10-01].

After a deploy, `check_published.py` walks the project page and a talk locally, every click and every scene of its
episodes, and requests each file they loaded from the published site [Yohaï, 2026-10-01]. Do not edit `site/` while
`shoot.py`, `stress.py` or `check_published.py` runs: the dev server reloads open pages on every change, which breaks
the run.

Every slide carries `data-status`: `ready`, `draft` (content in place, visuals or wording to finish) or
`aspirational` (shows what we want; the data or code behind it does not exist yet). Drafting views show it as a
corner badge; `?present` hides the badges. An aspirational slide describes its missing visual in a
`.placeholder` and has an entry in the talk's `PLAN.md`; status changes land in the same commit as the work.

## Writing slides

- Slide text comes from verified human material, in its words: the paper, the project page (`../index.html`: its
  headings and labels, such as "What's in an active-vision model?", "See it in action"), the README, the authors'
  own published writing, and the sources the paper cites. Look there for every title before writing one
  [Yohaï, 2026-10-01].
- Reuse human wordings, turns of phrase and framings verbatim where they fit; write new wording only when nothing
  human fits, and keep it plain. A paraphrase of a human sentence is not human wording. Read the paper section a slide
  covers before writing the slide [Yohaï, 2026-10-01].
- Titles take the written register: the page's and the paper's headings, the authors' written applications, or the
  plain name of what the slide shows. A spoken turn of phrase (a joke, an aside, a rhetorical question) is never a
  title [Yohaï, 2026-10-01].
- A title names the thing the slide shows ("Canvas Attention", "Scene-Relative RoPE", "A CanViT rollout") or
  states a claim the paper or a cited source makes ("No policy can make up for a poor observer"). Never a
  tagline, a metaphor or a slogan of your own.
- A title is literally true and matches what the slide shows. History is told as the paper tells it: active
  models "have struggled to match" passive ones.
- A title says what the slide shows or claims, in the audience's terms; never the name of an experimental condition
  [Yohaï, 2026-10-01].
- A setting detail (the glimpse size, a canvas grid) belongs in the body or the notes, never in a title or a
  diagram's caption [Yohaï, 2026-10-01].
- No circumlocutions [Yohaï, 2026-10-01].
- Labels are as short as the thing they name: "Paper", "Code", "Models". Install commands use uv:
  `uv add canvit-pytorch` [Yohaï, 2026-10-01].
- Never number slides, sections or items in outlines, plans and notes; refer to a slide by its title or its id
  [Yohaï, 2026-10-01]. Build states are named, never numbered.
- What a talk shows, and how many slides it takes, follows from what makes it correct, clear, meaningful and
  beautiful [Yohaï, 2026-10-01]; no slides-per-minute rule decides it.
- Layouts use the slide: the visual as large as the 1152 × 628 px content area allows, no large empty regions, the
  most important element the largest [Yohaï, 2026-10-01].
- The last slide stays up during questions: the links (QR codes, largest) and the funding acknowledgments on one
  slide [Yohaï, 2026-10-01].
- Visual storytelling first and always; less is more [Yohaï, 2026-10-01]. A slide states a fact once: no
  information repeated on the same slide. Check centering and balance on the screenshot: reveal.js sets
  `display: block` inline on the shown slide, so a section's own `display: flex` does not apply; center blocks with
  auto margins or a wrapper.
- Ask what the slide's takeaway is, then make it the most visible thing on the slide; nothing else gets more
  emphasis than it. A number the audience does not need goes to the notes, never into a large callout
  [Yohaï, 2026-10-01]. A paper figure that carries the point is shown as large as the slide allows, with annotations
  (rings, `.marks`) pointing at the part that matters.
- A description on a slide is precise and distinctive [Yohaï, 2026-10-01].
- No unnecessary or obvious caveats [Yohaï, 2026-10-01].
- Think about axis scalings, including transforms such as Fisher's [Yohaï, 2026-10-01].
- A number on a slide carries its unit [Yohaï, 2026-10-01].
- An example must be inspiring [Yohaï, 2026-10-01].
- Never overload a slide with text [Yohaï, 2026-10-01].
- Quality over quantity [Yohaï, 2026-10-01]. The deck holds only slides whose visual carries the point. A slide that
  would be a list of lines, a paragraph or a placeholder box stays out of `index.html` until its visual exists; its
  plan lives in `OUTLINE.md` and `PLAN.md`.
- The neuroscience is the paper's: human vision is active (gaze shifts toward regions of interest, sequential,
  with strategic planning, integration across time in visual working memory, and top-down recurrent
  feedback), each point with its citation.
- Under one title, direct attention in sequence rather than showing everything at once [Yohaï, 2026-10-01]: one
  element large while it is the subject, then smaller as the next arrives; arrows that draw themselves between what
  relates; fades between states; dynamic cutouts and zooms. A slide may open on the question or an overview (for the
  policies: a grid of each policy's color, name and purpose) and then fade into the animation. Use transitions,
  step-by-step reveals and highlights throughout. Built with named states: a fragment
  `<span class="fragment step-marker" data-shows="teacher">` gives its slide the class `shows-teacher` while it is
  shown, and the slide's CSS keys on those names (`deck.js`); `.draw` arrows (`deck.css`). Code is walked through the
  same way: a fragment with `data-lines="A-B"` highlights those lines of its slide's code block (`deck.js`).
- Be visually consistent: where CanViT appears in a diagram, use its logo, `../assets/logos/canvit-wordmark.svg`
  [Yohaï, 2026-10-01].
- Think about what must be introduced or highlighted when, as each concept first appears on screen or must be spoken
  [Yohaï, 2026-10-01].
- Think through when things update and how: a map that depends on a step updates with that step, without an extra
  click [Yohaï, 2026-10-01].
- Semantic segmentation needs an introduction for this kind of audience; the `#history` slide of the MAIN 2026 talk
  shows the kind wanted: an image, then its mask sliding across it [Yohaï, 2026-10-01]. `.wipe` (`deck.css`) reveals
  an overlay across its image. Overlays can confuse: also try side by side and one after the other.
- DINOv3 feature maps and probability maps are drawn with hard nearest-neighbor upsampling [Yohaï, 2026-10-01].
- Choose every example image because, out of the whole validation set, it illustrates the point best, found by a
  sweep that measures the point, and checked by eye; never because it is convenient or already in use
  [Yohaï, 2026-10-01]. Before the sweep, the slide's `OUTLINE.md` entry states the properties a good example must
  have (what must be visible, what the model must do, what the audience must recognize, what must not mislead), and
  the sweep measures them. Examples must show the work at its best and match what the slide means to say.
- Show what the model thinks of a few named classes (their probabilities or logits, two or three at a time) rather
  than a full segmentation map, whose class colors the audience cannot read [Yohaï, 2026-10-01].
- For any slide that matters, try several ways to present its point, render each (screenshot, or play the
  animation), compare what works and what does not, then integrate the best parts [Yohaï, 2026-10-01]. Variants are
  drafted outside the deck (an untracked page in the talk's directory, deleted after the choice) and the choice is
  recorded in `OUTLINE.md`.
- Demonstrate, then name [Yohaï, 2026-10-01]. The first idea for any claim is how to show it happening, on a real
  image, with the real model or the real baseline. A claim shown only as text or a diagram is a placeholder until
  such a demonstration exists or is ruled out.
- No subtext [Yohaï, 2026-10-01]: on-screen text is the title, the labels a visual cannot be read without, and the
  citation of another paper's figure. A caption that restates the visual, a takeaway line, a conditions line, and a
  citation of the authors' own paper go to the notes.
- Speaker notes (`<aside class="notes">`) are bullet points at most, without invented logical links, only where they
  help, and in the presenter's voice; when unsure, leave a point out rather than dress it up [Yohaï, 2026-10-01].
- Glimpse blue, canvas red and policy teal keep their meanings on every slide (`../css/canvit.css`).
- A figure from another paper is cited on the slide (`.cite`: authors, year, venue).
- Animations play while their slide is shown and pause when it is left (`data-play`); a slide that builds
  click by click uses fragments, so the replay button and the speaker view stay in step.
- Interactive figures (the live model, the charts' tooltips) must not advance the deck when clicked: they are
  listed in `INTERACTIVE` in `deck.js`, or carry `data-no-advance`.
