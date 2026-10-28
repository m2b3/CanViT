# The MAIN 2026 talk: how to work

How to work on the talk "Toward Active-Vision Foundation Models" (MAIN 2026, 2026-10-30). This guide holds rules
that stay true while the talk is made; the talk's positions, open questions and open work, which change as we
iterate, are in `TALK.md`. The repository's `AGENTS.md` and `site/AGENTS.md` apply too. `CLAUDE.md` is a symlink to
this file. Keep it current: when the workflow changes, change it here in the same commit, and improve the workflow
whenever a session shows it failing.

The talk is developed on the branch `talk-2026-10-30-MAIN`, which takes in `main` regularly; never on `main`. The
branch may hold messy working material; it merges into `main` only after the talk, as one squashed commit, once the
whole diff has been checked with the authors. Its own history does not matter. The repository is public: pushing the
branch publishes what it holds.

## Where knowledge lives

Each thing has one home; everything else points to it.

| File | Holds |
|---|---|
| `AGENTS.md` | How to work. |
| `TALK.md` | The authors' positions (marked (Y), dated), agents' inferences and proposals (marked (A)), open questions, the deck as it stands, open work. |
| `sources/checked-facts.md` | Facts about CanViT, its paper, its code and the talk's measurements, with where each lives. "Essential" first: what the current argument rests on; then the bank. |
| `sources/literature.md` | What other papers say, read in the papers, with pages, quotes and numbers. |
| `sources/*.json`, `sota-history.md` | Data the deck reads, each with its provenance. |
| `experiments/` | The experiments that compute the deck's data; `experiments/training_cost/README.md` documents the baselines' compute. |
| `assets/*/PROVENANCE.md` | Where each figure and logo comes from. |

- Facts and interpretation stay apart. A fact names its source and the date it was checked; an interpretation is
  marked as one, with whose it is. A subagent's reading is an inference until checked.
- The authors' formulations stay apart from agents' inferences. Positions are written in clean words, never as
  pasted quotes of the authors' messages; a formulation they gave for the talk is kept as given.
- A position is not a fact and a draft is not a decision. Nothing in the deck or the notes is approved unless the
  authors discussed it explicitly; the notes files are agents' working files, which the authors do not read.
- Remove what is stale in the same step that makes it stale. Text that oversimplifies, erases nuance or states an
  open question as settled misleads the next agent: fix it or delete it. Git keeps what is deleted.
- Keep hard-won facts (numbers read in a paper, a table and page, a measurement) even when no slide uses them now.
- Do not spread one subject over several files; a new file needs a reason that no existing home serves.
- Unpublished results beyond what the deck shows, private sources and anything about CanViT 2 beyond broad strokes
  stay out of the repository.

## Working with the authors

- Discussion is a quick back and forth: ask less, and more carefully. Stay at the level the discussion is at; when
  asked to zoom out, return to the overall message.
- Scope claims to what they cover: "a", not "the", where others exist; a partial scope is never stated as the whole.
  Keep distinct consequences distinct. No superlatives without the comparison that earns them, and no slogans.
- Think a causal claim through before stating it, and do not invent objections an expert would not raise.
- Take nothing at face value, including the authors' and agents' earlier statements; read and understand material
  fully before presenting it [Yohaï, 2026-10-01].
- A review re-derives the argument: what problem each section answers, what each slide must establish and why it comes
  where it does. Moving slides follows from that; it is not the review [Yohaï, 2026-10-27].

## Files and tools

- `../deck.js`, `../deck.css`: the deck every talk runs on (reveal.js at 1280 × 720, the page's light identity from
  `../../css/canvit.css`); `../deck.css` also holds the utilities
  every talk may use (`.draw` arrows, `.marks` rings, `.step-marker`). Slide conventions are documented at the top of `../deck.js`: `data-play`,
  `data-canvit-target`/`data-canvit-t`, `data-shows`, `<section data-status>`.
- `../package.json`: reveal.js, pinned and installed locally (`npm ci --prefix site/slides`), so a talk runs offline.
- `../shoot.py`: screenshots of every slide in Chromium, the browser errors and layout problems (below).
- `../stress.py`, `../check_published.py`: a talk skipped through at speed; its files checked against the published site
  (below).
- This directory: `index.html` (every slide, with speaker notes), `talk.css` (rules scoped by slide id), talk-only
  modules and assets, the notes listed under "Where knowledge lives", and `experiments/`: the experiments that
  compute the deck's data, a package per experiment, each step run from this directory as
  `uv run --no-sync --project ../../../canvit-pytorch python -u -m experiments.<experiment>.<step>`
  (`training_cost` in `../../../paper/exporter`'s environment). Sweeps write to `../../.experiments/` (ignored by
  git); the exports the deck loads go to `../../data/talk/` (deployed, never committed; `deploy.sh` ships the data
  files the deck names, never the experiments). `experiments/build_deck_data.sh` reruns them all, or the named ones.

Computing, exporting and plotting are separate steps [Yohaï, 2026-10-01]: an experiment writes its results as
uncolored data (labels, probabilities, scalar maps, metrics, viewpoints) to files; figures and slides read those files
and choose colors, colormaps and layout, so a figure can be redrawn differently without running the model again.
This is the web bundle's rule (`canvit-pytorch/docs/viz.md`, "Data layers are lossless and uncolored").

A visual that shows CanViT itself (recorded rollouts, features, the live model, the paper's results) is a
`<canvit-*>` component in `../../js/canvit/`, usable on the project page as well; a talk never keeps its own copy.
Talk-only visuals (a history timeline, a title slide) stay in the talk's directory. Data come from the same places
as the page's: bundles under `../../data/` recorded by `canvit_pytorch.viz`, the paper's exports under
`../../assets/paper/`, and numbers in `<span data-macro>` checked by `../../check_paper_numbers.py`. A number the paper
does not carry yet (a new experiment) has an `XXX` comment beside it naming its source.

## Paper figures

The paper's figures and tables can be tried on slides before the paper version they come from is public.
`sources/paper-figures.json` lists each one, one line per entry: a figure by its export name (`paper/exports/NAME.pdf`),
a table by its LaTeX label. An entry with `"enabled": true` is a slide after the slide whose id its `where` names, or
at the end of Backup for `"backup"`, headed by its `title` (HTML, none when empty), with its `notes` (a list of HTML
bullets) as speaker notes and its `status` as `data-status` (draft when absent); edit the file and the open deck
reloads with the slide added or removed. The slides follow "Writing slides" below: the paper's caption is not shown,
so what it says goes in the title or the notes. `../paper_figures.py` does the rest (its docstring has the details);
`$PAPER` is a checkout holding the paper version to show:

```bash
TALK=2026-10-30-Montreal-AI-and-Neuroscience-MAIN
uv run site/slides/paper_figures.py try --talk $TALK --paper $PAPER learned_policies_miou --where learned-policy
uv run site/slides/paper_figures.py accept --talk $TALK learned_policies_miou
uv run site/slides/paper_figures.py reject --talk $TALK learned_policies_miou
```

- Trying stays local. `try` renders the entries from a commit of `$PAPER` (`--rev`, default `HEAD`) into
  `.paper-figures/`, which git ignores and `deploy.sh` never ships, and enables them; each slide is marked as a trial
  until accepted. A figure is its export as SVG; a table is cut from the camera-ready build at the same commit, from
  its top rule to its bottom rule.
- Accepting publishes. `accept` moves the trial render into `assets/paper-figures/` and records its commit, source,
  renderer and checksum in the entry; commit the render and the manifest together. The branch is public; the authors
  accept the camera-ready's figures and tables on it. A figure that must build up click by click is made from its own
  code instead, never by drawing over a render (`#canvit-rollout`: `experiments/canvit_rollout/`).
- Rejecting leaves no trace. `reject` deletes the entry's renders, trial and accepted, and resets its line as `sync`
  wrote it. Setting `"enabled": false` only hides the slide.
- `deploy.sh` runs `paper_figures.py check`, which refuses an enabled entry that is not accepted, an accepted render
  that differs from its record, a render that no entry accepts, and a `where` that names no slide.
- `paper_figures.py sync --talk $TALK --paper $PAPER` adds the figures and tables the paper has gained.

## Workflow

Before any work on the talk, and again after every context reset or compaction, read the paper
(`../../../paper/latex/CanViT_Toward_AVFMs.tex`) in full, appendix included, and `TALK.md`. Summaries and memory
of them are not a substitute [Yohaï, 2026-10-24].

In every session, and again in every context window, view every main figure of the paper (the figures of its body,
`../../../paper/latex/figures/exported/`: `canvit_overview.pdf`, `arch_overview.pdf`,
`canvas_attention_combined.pdf`, `main_results.pdf`, `resolution_and_mask_size.pdf`) [Yohaï, 2026-10-25].

[Yohaï, 2026-10-01] A talk is designed in `TALK.md` ("The deck as it stands") before it becomes slides: for every slide, the
proposition it states or proves, what is on screen, the visual and how it builds, what is said, the
sources and the status. Each entry separates what the slide must show, say or carry across (without it the story
breaks or a claim goes unsupported) from what would be interesting or good to show; both get made, and the first
is never cut for the second. The outline is reviewed (independent reviewers: fresh subagents, `codex exec` with model
`gpt-6-astra`, whose sandbox reaches the network only with `-c sandbox_workspace_write.network_access=true`, asked
open questions about understanding, key messages and what they would do) and revised until the authors are confident
in it; slides are built from it, and an outline change comes before a slide change.

```bash
npm ci --prefix site/slides                     # once: reveal.js
node site/serve.mjs 8765                        # serve site/; open pages reload when a file changes
# http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/   (S: speaker view with notes; with a second
#   tab of the deck open, the speaker view stops following the slides)
uv run site/slides/shoot.py --url http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/ \
    --out site/.screens/main-2026 > site/.screens/shoot.log 2>&1
uv run --no-sync --project canvit-pytorch python site/check_paper_numbers.py
```

Slide preparation keeps nuance and ambiguity open rather than collapsing them early, gathers primary material, and
makes quick back-and-forth on views into the data easy [Yohaï, 2026-10-01].

Every request of the authors that is not yet done goes into `TALK.md` ("Open work") the moment it
arrives, stated plainly, before any work on it, and leaves only when it is done; requests survive the end of a
session only there [Yohaï, 2026-10-01].

Every fact the talk needed checked in the paper, the code or the data is noted in `sources/checked-facts.md`
with where it lives, the same day [Yohaï, 2026-10-01], so the next session finds it without rereading.
A note points to its source (paper section, module and symbol, file path) and states the fact; numbers the paper
generates stay in its macros.

`../shoot.py` exits nonzero when the page logged an error; read `errors.txt`, `layout.txt` (titles that wrap, content in
the footer band, a word alone on the last line of a text block, text blocks that overlap, speaker notes whose
[CLICK] marks do not match the slide's clicks) and the images of every slide you touched, with all fragments shown (default) and before them
(`--first-fragment`), or click by click (`--steps`).
Screenshots are review material and are never committed (`site/.screens/` is ignored).

`../stress.py` skips through a talk quickly, forward, back and at random, and checks that each slide lands in the state
a settled visit gives it, with nothing playing off its slide [Yohaï, 2026-10-01].

The published deck has no Backup slides (`../published_deck.py` cuts them), so the page's script reaches a Backup slide
only through `drawIfPresent`; the cut refuses a deck that looks one up directly, which would not start.

After a deploy, `../check_published.py` walks the project page and a talk locally, every click and every scene of its
episodes, and requests each file they loaded from the published site [Yohaï, 2026-10-01]. Do not edit `site/` while
`../shoot.py`, `../stress.py` or `../check_published.py` runs: the dev server reloads open pages on every change, which breaks
the run.

Every slide carries `data-status`: `ready`, `draft` (content in place, visuals or wording to finish) or
`aspirational` (shows what we want; the data or code behind it does not exist yet). The slides show no badge for it.
An aspirational slide describes its missing visual in a `.placeholder` and has an entry in `TALK.md`; status changes
land in the same commit as the work.

## Writing slides

- Slide text comes from verified human material, in its words: the paper, the project page (`../../index.html`: its
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
- Correct names are paramount. A title describes what the slide shows, never a compressed label that goes stale or
  misdescribes it, and fits on one line: what else must come across is told visually or said by the speaker
  [Yohaï, 2026-10-24]. No colon formulas that pack a definition or claim into a title. A slide's id names what the
  slide is, not its example.
- A title is literally true and matches what the slide shows. History is told as the paper tells it: active
  models "have struggled to match" passive ones.
- A title says what the slide shows or claims, in the audience's terms; never the name of an experimental condition
  [Yohaï, 2026-10-01].
- A setting detail (the glimpse size, a canvas grid) belongs in the body or the notes, never in a title or a
  diagram's caption [Yohaï, 2026-10-01].
- No circumlocutions [Yohaï, 2026-10-01].
- No middle dot (·) on a slide: reaching for one to join items means the slide says too much; cut to one item or give
  each its own place [Yohaï, 2026-10-25].
- No counted parallelism ("one X, one Y, seven Z"), in titles or anywhere else (the repository's `AGENTS.md`).
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
- A curve keeps its style (solid or dashed) across a slide's steps [Yohaï, 2026-10-25].
- A number on a slide carries its unit [Yohaï, 2026-10-01].
- An example must be inspiring [Yohaï, 2026-10-01].
- Never overload a slide with text [Yohaï, 2026-10-01]. A slide says exactly what is needed and nothing more;
  technical details that the audience can find in the paper stay off it [Yohaï, 2026-10-24].
- The talk never repeats itself: each result appears once, where it serves the argument [Yohaï, 2026-10-24].
- Quality over quantity [Yohaï, 2026-10-01]. The deck holds only slides whose visual carries the point. A slide that
  would be a list of lines, a paragraph or a placeholder box stays out of `index.html` until its visual exists; its
  plan lives in `TALK.md`.
- The neuroscience is the paper's: human vision is active (gaze shifts toward regions of interest, sequential,
  with strategic planning, integration across time in visual working memory, and top-down recurrent
  feedback), each point with its citation.
- Under one title, direct attention in sequence rather than showing everything at once [Yohaï, 2026-10-01]: one
  element large while it is the subject, then smaller as the next arrives; arrows that draw themselves between what
  relates; fades between states; dynamic cutouts and zooms. A slide may open on the question or an overview (for the
  policies: a grid of each policy's color, name and purpose) and then fade into the animation. Use transitions,
  step-by-step reveals and highlights throughout. Built with named states: a fragment
  `<span class="fragment step-marker" data-shows="teacher">` gives its slide the class `shows-teacher` while it is
  shown, and the slide's CSS keys on those names (`../deck.js`); `.draw` arrows (`../deck.css`). Code is walked through the
  same way: a fragment with `data-lines="A-B"` highlights those lines of its slide's code block (`../deck.js`).
- Be visually consistent: where CanViT appears in a diagram, use its logo, `../../assets/logos/canvit-wordmark.svg`
  [Yohaï, 2026-10-01].
- Think about what must be introduced or highlighted when, as each concept first appears on screen or must be spoken
  [Yohaï, 2026-10-01].
- Think through when things update and how: a map that depends on a step updates with that step, without an extra
  click [Yohaï, 2026-10-01].
- Semantic segmentation needs an introduction for this kind of audience: an image, then its mask sliding across it,
  as `#gap` does [Yohaï, 2026-10-01]. `.wipe` (`../deck.css`) reveals
  an overlay across its image. Overlays can confuse: also try side by side and one after the other.
- DINOv3 feature maps and probability maps are drawn with hard nearest-neighbor upsampling [Yohaï, 2026-10-01].
- A glimpse shown on a slide is what the model receives: 128 × 128 px, sampled from the 512 px scene with
  `sample_at_viewpoint` (bilinear, no antialiasing), never a crop of the scene at its own resolution
  [Yohaï, 2026-10-25].
- Choose every example image because, out of the whole validation set, it illustrates the point best, found by a
  sweep that measures the point, and checked by eye; never because it is convenient or already in use
  [Yohaï, 2026-10-01]. Before the sweep, the slide's entry in `TALK.md` states the properties a good example must
  have (what must be visible, what the model must do, what the audience must recognize, what must not mislead), and
  the sweep measures them. Examples must show the work at its best and match what the slide means to say.
- Show what the model thinks of a few named classes (their probabilities or logits, two or three at a time) rather
  than a full segmentation map, whose class colors the audience cannot read [Yohaï, 2026-10-01].
- For any slide that matters, try several ways to present its point, render each (screenshot, or play the
  animation), compare what works and what does not, then integrate the best parts [Yohaï, 2026-10-01]. Variants are
  drafted outside the deck (an untracked page in the talk's directory, deleted after the choice) and the choice is
  recorded in `TALK.md`.
- How CanViT is trained (reinforcement learning or not, and other training details) is Yohaï's to present; agents do
  not put training details on slides that define what an Active-Vision Foundation Model is [Yohaï, 2026-10-25].
- Do what is right, not merely what is easy [Yohaï, 2026-10-24]: decide what matters and what would be good, then
  build it. Never assemble a slide from the ingredients at hand (existing components, data already recorded, a
  figure that happens to exist) because they are at hand.
- Do not smuggle information [Yohaï, 2026-10-24]: every concept, term or claim the argument relies on gets its own
  visible moment, on screen and said, never slipped into a caption, a heading, a thumbnail or the notes.
- Short signposting slides make each move of the argument explicit: a few words, a few arrows, 5 to 10 seconds,
  highlighting what matters [Yohaï, 2026-10-24]. The reasoning a signpost marks is said, never added to it as text
  [Yohaï, 2026-10-27].
- Demonstrate, then name [Yohaï, 2026-10-01]. The first idea for any claim is how to show it happening, on a real
  image, with the real model or the real baseline. A claim shown only as text or a diagram is a placeholder until
  such a demonstration exists or is ruled out.
- No subtext [Yohaï, 2026-10-01]: on-screen text is the title, the labels a visual cannot be read without, and the
  citation of another paper's figure. A caption that restates the visual, a takeaway line, a conditions line, and a
  citation of the authors' own paper go to the notes.
- Speaker notes (`<aside class="notes">`) are many short bullet points, never paragraphs [Yohaï, 2026-10-25],
  without invented logical links, only where they help, and in the presenter's voice; when unsure, leave a point out rather than dress it up [Yohaï, 2026-10-01].
- Speaker notes are written the way their speaker says them, conversationally. A note that cannot be made to flow
  means its content does not hold together: stop, think it through and fix the content, not the wording
  [Yohaï, 2026-10-24].
- The speakers say what the notes say: a connector they need ("and", "conversely") is in the notes, and nothing is
  added on stage. Notes have no transition lines [Yohaï, 2026-10-25].
- Connect points only where the logic connects them. A fact can be set up on its own and connected later, as the
  paper's introduction does [Yohaï, 2026-10-25].
- Each beat of a slide starts a new bullet, with a click between beats. Bullets are a breath long, with the key
  logical connectors and terms in bold. Anything said for more than a few seconds has a visual on the slide
  [Yohaï, 2026-10-25].
- "We" means the authors. A pointer ("this", "that") refers to the thing just named; when its referent is not
  immediate, restate the thing ("drawing inspiration from human vision", not "from this process") [Yohaï, 2026-10-25].
- Speaker notes an agent drafted start with `[AI draft: not the authors' text]` until an author rewrites them; an
  agent's text kept beside an author's is marked the same way. Notes taken from an author's run-through start with
  a line naming the run-through and keep its wording, false starts removed and self-corrections applied. Agents' framings and words ("a twist") are never passed
  off as the authors' [Yohaï, 2026-10-25].
- An agent's notes for a slide are not acceptable until it has answered each of these in writing for that slide, the
  way the authors work through a slide together, which takes them fifteen to twenty minutes [Yohaï, 2026-10-25]:
  - What the slide must make the listener understand, and whether the first bullet says it, not what is visible.
  - What the listener knows at this point of the talk: every term used is defined earlier or here; a point made
    earlier returns only as a callback that moves the argument.
  - For each state of the slide (before any click, then after each), what appears and what is said with it.
  - Which sentence of the paper, or of the cited source, already says it; its wording and logical connectors are
    reused where they fit.
  - Each claim checked against the paper or the data, at its scope: what CanViT does is not stated as what every model
    does, and no claim goes beyond its evidence.
  - Every word read as a listener who sat through the slides before: no narration of what they can see, no colon
    formula packing a label and its definition, no pointer without an immediate referent, no undefined jargon, no
    invented connectors.
  - Spoken length at the authors' measured speaking rate (`TALK.md`, rehearsal timings).
  `../shoot.py` checks the mechanical part (clicks against [CLICK] marks, and Transition lines in AI drafts); passing
  it is necessary, never sufficient.
- Glimpse blue, canvas red and policy teal keep their meanings on every slide (`../../css/canvit.css`).
- Viewing policies keep the paper's figure colors (`../../js/canvit/policies.js`) [Yohaï, 2026-10-25].
- A figure from another paper is cited on the slide (`.cite`: authors, year, venue).
- Animations play while their slide is shown and pause when it is left (`data-play`); a slide that builds
  click by click uses fragments, so the replay button and the speaker view stay in step.
- Interactive figures (the live model, the charts' tooltips) must not advance the deck when clicked: they are
  listed in `INTERACTIVE` in `../deck.js`, or carry `data-no-advance`.
