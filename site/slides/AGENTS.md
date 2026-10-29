# Talks

How to make a talk on the deck in this directory. The repository's `AGENTS.md` and `site/AGENTS.md` apply too; each
talk's directory has its own `AGENTS.md` for that talk. Keep this guide current: when the workflow changes, change it
here in the same commit.

## Files

- `deck.js`, `deck.css`: the deck every talk runs on (reveal.js at 1280 × 720, the project page's identity from
  `../css/canvit.css`). `deck.js` documents the slide conventions at its top: `data-shows`, `data-play` and
  `data-stills`, `data-canvit-target`, `data-lines`, `data-status`. `deck.css` holds the utilities every talk may use
  (`.draw` arrows, `.marks` rings, `.wipe`, `.step-marker`).
- `package.json`: reveal.js, pinned and installed locally (`npm ci --prefix site/slides`), so a talk runs offline.
- `shoot.py`: screenshots of every slide in Chromium, with the browser's errors and layout problems.
- `stress.py`: a talk skipped through at speed, forward, back and at random; each slide must land in the state a
  settled visit gives it, with nothing playing off its slide.
- `check_published.py`: after a deploy, walks a talk locally, every click and every scene of its episodes, and
  requests each file it loaded from the published site.
- `pdf_backup.py`: every click of every slide as a page (`slides.pdf`), the same pages as `slides.pptx` with each
  click's notes, and a presenter's copy with each slide's last state and notes (`notes.pdf`).
- `single_file.py`: the deck as one HTML file that runs from disk with no server and no network, checked by opening
  it offline and walking every click.
- `export_talk.py`: the single file, the PDF and the PowerPoint under one name, for an organizer's shared folder.
- `paper_figures.py`, `paper-figures.js`: the paper's figures and tables as slides (below).
- `YYYY-MM-DD-Event/`: a talk: `index.html` (every slide, with speaker notes), `talk.css` (rules scoped by slide id),
  talk-only modules and assets, `TALK.md` (the talk's design and the authors' positions), `sources/` (the material it
  draws on, each with its provenance) and `experiments/` (what computes its data).

The fallbacks on the day, in order: the published deck; the deck served from a laptop; the single file;
`slides.pptx`; `slides.pdf`. Rebuild the exports after the last change.

## Data and visuals

Computing, exporting and plotting are separate steps: an experiment writes its results as uncolored data (labels,
probabilities, scalar maps, metrics, viewpoints); figures and slides read those files and choose colors, colormaps
and layout, so a figure can be redrawn without running the model again (`canvit-pytorch/docs/viz.md`, "Data layers
are lossless and uncolored"). A talk's experiments write to `../data/talk/`, which is deployed and never committed;
`deploy.sh` ships the data files a deck names, never the experiments.

A visual that shows CanViT itself (recorded rollouts, features, the live model, the paper's results) is a `<canvit-*>`
component in `../js/canvit/`, usable on the project page as well; a talk never keeps its own copy. Talk-only visuals
stay in the talk's directory. Data come from the page's sources: bundles under `../data/` recorded by
`canvit_pytorch.viz`, the paper's exports under `../assets/paper/`, and numbers in `<span data-macro>` checked by
`../check_paper_numbers.py`. A number the paper does not carry has an `XXX` comment beside it naming its source.

## Paper figures

A talk's `sources/paper-figures.json` lists the paper's figures (by export name, `paper/exports/NAME.pdf`) and tables
(by LaTeX label). An enabled entry is a slide after the slide its `where` names (or at the end of the appendix),
headed by its `title`, with its `notes` and `status`. The paper's caption is not shown. `paper_figures.py` documents
the rest:

```bash
uv run site/slides/paper_figures.py try --talk $TALK --paper $PAPER NAME --where SLIDE  # local, never deployed
uv run site/slides/paper_figures.py accept --talk $TALK NAME                           # publishes the render
uv run site/slides/paper_figures.py reject --talk $TALK NAME                           # leaves no trace
uv run site/slides/paper_figures.py sync --talk $TALK --paper $PAPER                   # adds new figures and tables
```

`deploy.sh` runs `paper_figures.py check`. A figure that must build up click by click is made from its own code,
never by drawing over a render.

## Workflow

A talk is designed in its `TALK.md` before it becomes slides: for every slide, the proposition it states or proves,
what is on screen, how it builds, what is said, its sources and its status. What a slide must show, say or carry
across is kept apart from what would be good to show; both get made, and the first is never cut for the second. The
outline is reviewed by independent readers (fresh agents, asked open questions about understanding and key messages)
and revised until the authors are confident in it; an outline change comes before a slide change.

Open work (the authors' requests not yet done, open questions, plans) stays out of the repository, in the agent's
private memory, and leaves when it is done. Every fact checked in the paper, the code or the data goes into the talk's
`sources/checked-facts.md` the same day, with where it lives.

```bash
npm ci --prefix site/slides      # once: reveal.js
node site/serve.mjs 8765         # serves site/; open pages reload when a file changes
uv run site/slides/shoot.py --url http://127.0.0.1:8765/slides/$TALK/ --out site/.screens/$TALK
uv run site/slides/stress.py --url http://127.0.0.1:8765/slides/$TALK/
uv run --no-sync --project canvit-pytorch python site/check_paper_numbers.py
```

`shoot.py` exits nonzero when the page logged an error. Read `errors.txt`, `layout.txt` (titles that wrap, content in
the footer band, a word alone on a line, overlapping text, speaker notes whose [CLICK] marks do not match the clicks)
and the images of every slide you touched: all fragments shown (default), before them (`--first-fragment`), or click
by click (`--steps`). Screenshots are never committed (`site/.screens/` is ignored).

The dev server reloads open pages on every change under `site/`, which breaks a run: do not edit `site/`, deploy, or
run another capture while `shoot.py`, `stress.py`, `check_published.py` or an export runs.

Every slide carries `data-status`: `ready`, `draft` (content in place, visuals or wording to finish) or
`aspirational` (the data or code behind it does not exist yet, described in a `.placeholder` and in `TALK.md`).

Every slide is published, the appendix's included; the page's script draws a slide through `drawIfPresent`, so a
slide can leave the deck before its drawing code does.

## Writing slides

What a talk shows, and how many slides it takes, follows from what makes it correct, clear, meaningful and beautiful;
no slides-per-minute rule decides it. The last slide stays up during questions: the links (QR codes, largest) and the
funding acknowledgments.

Titles and text:

- Correct names are paramount.
- Slide text comes from verified human material, in its words: the paper, the project page, the README, the authors'
  published writing, the sources the paper cites. Reuse their wordings verbatim where they fit; a paraphrase is not
  human wording. Read the paper section a slide covers before writing the slide.
- A title names the thing the slide shows or states a claim the paper or a cited source makes, in the written
  register and the audience's terms. It is literally true, fits on one line, and is never a tagline, metaphor,
  slogan, spoken aside, colon formula or the name of an experimental condition. A setting detail (glimpse size,
  canvas grid) belongs in the notes.
- History is told as the paper tells it: active models "have struggled to match" passive ones.
- No circumlocutions, no middle dot (·) joining items, no counted parallelism, no unnecessary caveats.
- Labels are as short as the thing they name: "Paper", "Code", "Models". Install commands use uv.
- Never number slides, sections or items; refer to a slide by its title or id. Build states are named.
- No subtext: on-screen text is the title, the labels a visual cannot be read without, and the citation of another
  paper's figure (`.cite`: authors, year, venue). Captions, takeaways, conditions and citations of the authors' own
  paper go to the notes.
- A number on a slide carries its unit. A description is precise and distinctive.

Visuals:

- Visual storytelling first; less is more. The deck holds only slides whose visual carries the point; a slide that
  would be a list, a paragraph or a placeholder stays in `TALK.md` until its visual exists.
- Ask what the slide's takeaway is and make it the most visible thing on it. A number the audience does not need goes
  to the notes. A paper figure that carries the point is as large as the slide allows, with rings (`.marks`) on the
  part that matters.
- Layouts use the slide: the visual as large as the 1152 × 628 px content area allows, no large empty regions, margins
  balanced. reveal.js sets `display: block` inline on the shown slide: center with auto margins or a wrapper, and check
  centering on the screenshot.
- A slide states a fact once; the talk never repeats a result. Each result appears where it serves the argument.
- Never overload a slide with text; technical details the audience can find in the paper stay off it.
- Direct attention in sequence: one element large while it is the subject, then smaller as the next arrives; arrows
  that draw themselves; fades between states; cutouts and zooms. Build with named states (`data-shows`) and fragments,
  so the replay button and the speaker view stay in step; walk code with `data-lines`.
- Think through what is introduced when, and when things update: a map that depends on a step updates with that
  step, without an extra click.
- Demonstrate, then name: show a claim happening on a real image with the real model or baseline. A claim shown only
  as text or a diagram is a placeholder until such a demonstration exists or is ruled out.
- Do not smuggle information: every concept, term or claim the argument relies on gets its own visible moment, on
  screen and said.
- Short signposting slides mark each move of the argument: a few words, 5 to 10 seconds. The reasoning a signpost
  marks is said, never written on it.
- For a slide that matters, try several presentations, render each, compare, and integrate the best parts. Variants
  are drafted outside the deck and the choice is recorded in `TALK.md`.
- Build what is right, not what is at hand: never assemble a slide from existing components or data because they
  exist.
- Visual consistency: CanViT's logo (`../assets/logos/canvit-wordmark.svg`) where CanViT appears in a diagram; glimpse
  blue, canvas red and policy teal keep their meanings (`../css/canvit.css`); viewing policies keep the paper's colors
  (`../js/canvit/policies.js`); a curve keeps its style across a slide's steps.
- Think about axis scalings, including transforms such as Fisher's.
- Animations play while their slide is shown and pause when it is left (`data-play`). Interactive figures must not
  advance the deck when clicked (`INTERACTIVE` in `deck.js`, or `data-no-advance`).

Examples and model outputs:

- Every example image is chosen because, out of the whole validation set, it illustrates the point best, found by a
  sweep that measures the point and checked by eye. Before the sweep, the slide's entry in `TALK.md` states what a
  good example must show; examples show the work at its best.
- An example must be inspiring.
- A glimpse shown on a slide is what the model receives: 128 × 128 px sampled from the 512 px scene with
  `sample_at_viewpoint`, never a crop at the scene's resolution.
- DINOv3 feature maps and probability maps are drawn with nearest-neighbor upsampling.
- Show the model's view of a few named classes (two or three at a time), not a full segmentation map whose colors the
  audience cannot read.
- Semantic segmentation needs an introduction: an image, then its mask sliding across it (`.wipe`). Overlays can
  confuse: also try side by side and one after the other.
- The neuroscience is the paper's: human vision is active (gaze shifts toward regions of interest, sequential, with
  strategic planning, integration across time in visual working memory, top-down recurrent feedback), each point with
  its citation.

Speaker notes (`<aside class="notes">`):

- Many short bullets, a breath long, in the speaker's voice, written the way they say it; key connectors and terms in
  bold. Each beat starts a new bullet, with a click between beats; anything said for more than a few seconds has a
  visual.
- They hold only what is said, and [CLICK]: no transition lines, no stage directions. The speakers say what the notes
  say; a connector they need is in the notes.
- Connect points only where the logic connects them. A note that cannot be made to flow means its content does not
  hold together: fix the content, not the wording.
- "We" means the authors. A pointer ("this", "that") refers to the thing just named; otherwise restate the thing.
- When unsure, leave a point out rather than dress it up. Agents' framings are never passed off as the authors':
  who wrote each slide's notes is recorded in `TALK.md`.
- Before an agent's notes for a slide are acceptable, it answers in writing: what the slide must make the listener
  understand, and whether the first bullet says it; what the listener knows at this point (every term defined, a
  point made earlier returning only as a callback that moves the argument); what appears and is said in each state;
  which sentence of the paper or source already says it; each claim checked at its scope; every word read as a
  listener (no narration of what is visible, no undefined jargon, no invented connectors); its spoken length at the
  speakers' measured rate. `shoot.py` checks the mechanical part; passing it is necessary, never sufficient.
