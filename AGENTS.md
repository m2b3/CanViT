# CanViT: guide for contributors and coding agents

`CLAUDE.md` is a symlink to this file.

Build software that is beautiful to read, difficult to misuse and easy to build
upon. The README is the product contract: what CanViT is, how to use it, and
what it reproduces. This file holds enduring engineering principles and the
conventions that are easy to get wrong, not work status or a catalog of
implementation details.

Coding agents do much of the development here and are among the code's main
readers. Design the tree, the APIs and the commands for how agents read, edit
and operate software.

## Layout: one home per thing

- `canvit-pytorch/`: the Python distribution `canvit-pytorch` (import
  `canvit_pytorch`). The model lives at the package top level; subsystems are
  verbs: `canvit_pytorch.pretrain`, `.specialize` (probes, fine-tuning),
  `.evaluate` (benchmarks), `.viz` (recorded rollouts for the web). Docs in
  `canvit-pytorch/docs/`; SLURM job scripts in `canvit-pytorch/slurm/`.
- `canvit-pytorch/tpu/`: a separate uv environment for ImageNet-1k
  fine-tuning on Cloud TPU (exact torch/torch_xla pins).
- `site/`: the project page, deployed to https://m2b3.github.io/CanViT/ by
  `.github/workflows/pages.yml` on pushes to `main` that touch it.
- `.github/workflows/release.yml`: PyPI release of `canvit-pytorch` on `v*`
  tags.

A fact, constant or URL lives in one place and everything else points to it.

Place and name things by meaning, never by convenience [author, 2026-09-26].
A fact's home is the module whose name says what the fact is about: the path
a reader who has never seen the code would guess before searching. When no
such module exists, create it. The convenient alternatives (the module the
consumers already import, the file that needs no new import, the class that
happens to be nearby) make the edit easy and the design wrong, and every
later reader pays for it. If the reason for a placement, a name or an API is
about effort or about the current import graph ("already imported",
"closest", "avoids a new file"), the reason is the defect: stop and find the
right home.

Facts about a component (a policy's abbreviation and description, a
checkpoint's geometry) live in the component's own definition, and consumers
read them from there. A dict keyed by component names in some other module is
a second copy that goes stale without an error [author, 2026-09-26]. The same
holds for defaults: one declaration, imported everywhere else.

The code serves the paper, its ablations, the released checkpoints and the
demos built on them. Code none of those need goes; git history keeps it
[author, 2026-09-26].

## Design

Aim for code that is obviously correct from its structure, not merely code
with no obvious defects. Types, names and assertions should carry as much of
the correctness argument as possible; review is the last line of defense.
Write for capable engineers: simplicity means fewer concepts and fewer ways to
get something wrong, not avoiding powerful language features.

- Seek the abstraction that makes a whole class of code unnecessary. Judge it
  by the code it eliminates and the future changes it makes straightforward.
  Avoid both repetitive plumbing and speculative generality.
- Challenge necessity first. Prefer deleting over simplifying, simplifying
  over optimizing, and optimizing over automating. All else equal, less code
  is better; the structure that remains should be beautifully designed
  [author, 2026-09-26]. Leave sound code alone.
- Make invalid states unrepresentable: frozen dataclasses whose construction
  establishes invariants, `Literal` or enums for closed vocabularies,
  jaxtyping or asserted shapes for tensors, and the same checks when loading
  configs and checkpoints.
- Be suspicious of boolean switches, optional fields that only matter
  together, and modes that multiply branches. Model distinct alternatives as
  alternatives; do not force facts that coexist into an either-or type.
- Match closed vocabularies exhaustively (`match` with `assert_never`) so a
  new case forces a decision. Put shared behavior with the type that owns it
  instead of repeating the match in every consumer.
- Check whether a constraint is self-imposed before building a workaround.
  Fix the design that produces recurring defects, not only their symptoms.
- One canonical path through each operation. Every option, fallback and
  compatibility layer needs a concrete reason to exist. Missing or unexpected
  input fails loudly; no silent defaults.
- Generated code, tables and documentation derive from authoritative
  definitions, never from brittle parsing or another hand-maintained copy.

## Names and structure

A misleading name, type, module boundary or nesting is a defect. Fix it
rather than teaching readers to remember the exception.

- Name concepts precisely and judge a name by its full import path. Brevity
  must not erase meaning; spelled-out names beat paper symbols.
- Make ownership and dependencies clear from the file tree and public APIs.
  Group by concept and nest subpackages as the concepts require; short files
  are fine [author, 2026-09-26]. Do not split or combine files merely to meet
  a size or directory convention.
- Everything sits precisely where it belongs, at whatever depth that takes,
  not where it is convenient to put it. A planned module tree is provisional:
  when understanding improves, move things and update the plan
  [author, 2026-09-26].
- The import graph is part of the architecture. The core model depends on no
  subsystem; move a misplaced responsibility rather than reaching into another
  package's internals.
- Define record vocabularies precisely (checkpoint configs, Hub model cards,
  web bundle manifests): what each field means, who writes it, and under what
  conditions.

## Economy and clarity

Codebase size matters, including the tokens needed to understand it. Every
line should earn its place; compactness comes from good design, not cryptic
spelling.

- Measure reading cost on production source, tests and maintained docs
  separately (`tokei -f`), excluding lockfiles and generated copies, and use
  it to find unnecessary concepts and repetition.
- Each fact has one authoritative home. Deduplicate concepts as well as text;
  finding differently worded copies requires reading, not only searching. A
  routine change should not require synchronized edits across the tree.
- Delete ceremony, redundant wrappers, obsolete paths and unnecessary
  comments. Comments explain a non-obvious reason, invariant or limitation.
  If an explanation needs many caveats, reconsider the design first.
- Write plain, direct prose. Fix false claims immediately. Avoid unsupported
  guarantees, invented distinctions, stale counts and copied defaults. Keep
  historical explanations in commit messages.
- Flag doubts and unresolved issues with a localized `XXX`, `TODO` or `FIXME`
  beside the relevant code, saying what the concern is.

## Agent interaction

- Make the file tree, module graph, names and types teach the design. A
  reader should find an operation's owner and correctness argument without
  loading the whole repository.
- Command-line entry points are tyro dataclass configs, so help text and
  defaults come from the definitions that drive execution. Results go to
  files or stdout; diagnostics go to logging.
- Errors name the operation, the affected identity (path, checkpoint, step)
  and the observed evidence.
- Put durable learning in its home: an invariant enforced in code, this
  guide, or a `TODO` beside its owner. Avoid parallel manuals and handoff
  narratives.
- Test interfaces as a user meets them: a fresh process, the installed
  package outside this checkout, released checkpoints from the Hub.

## Working method

- Start by reading the architecture: `uv run pypatree`, then the code. Reading
  is the first resort, not a response to getting stuck.
- Before changing a workflow, trace its entry points, callers, ownership and
  failure paths, and read the surrounding code in full. A symbol search does
  not replace reading.
- Trust no claim merely because it appears in code, documentation, an agent
  summary or this file. Verify important assumptions against the
  implementation and evidence. Check pinned dependency source or official
  documentation for external behavior.
- Distinguish observations, guarantees and hypotheses. A log describes one
  run, not universal behavior.
- Favor decisive simplification over patches around a bad abstraction.
  Rename, reorganize or rewrite when justified, in small coherent commits.
- Compatibility is a requirement to justify. Released checkpoints on the
  Hugging Face Hub have external readers: a change to model configs or
  state-dict keys keeps them loading, verified by loading them.
- Judge everything by leverage per line, tests included [author,
  2026-09-26]. A test earns its place by catching a plausible defect that
  nothing else would: a paper invariant, a numerical equivalence, a released
  checkpoint that must keep loading. Vacuous tests (restating the code) and
  brittle ones (pinning internals) go.
- Record enduring feedback here at the level it was given; keep specific
  constraints beside the code they constrain.

## Commands

From `canvit-pytorch/`:

```bash
uv sync --all-extras   # a plain `uv sync` is exact and removes the extras
uv run pypatree        # module tree, a good first look
uv run just            # lint, typecheck, test
```

## Conventions that are easy to get wrong

- Viewpoint centers are `(row, col)` in `[-1, 1]`, matching tensor indexing,
  not Cartesian `(x, y)`; scale `s` is the crop's half side, so a glimpse
  covers `s²` of the scene.
- Standard canvas grid: 32×32 tokens; patch size 16 px; glimpses 128 px.
- `torch.compile`: call `model(x)`, never `model.forward(x)`, which bypasses
  the compiled wrapper.
- Numbers reported anywhere (README, site, papers) come from saved evaluation
  outputs, never typed by hand.

## Git

Never `git add -A` or `git add .`; stage files by name and read the staged
diff before committing.
