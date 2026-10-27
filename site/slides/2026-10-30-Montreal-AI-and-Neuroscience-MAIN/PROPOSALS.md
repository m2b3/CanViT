# Proposals from the camera-ready's additions

Agent proposals (2026-10-26) for the authors' review of Part II: changes larger than a correction, each with where it
goes, what changes and why. Paper locations are the camera-ready's LaTeX labels and exports, which reach this branch
when the camera-ready does; values stay in those exports. A decided entry moves into the deck or into `TALK.md`, and
this file goes once it is empty.

## Learning where to look (`#learned-policy`)

**Show both learned objectives.** The camera-ready's figure (`fig:learned-policies`) and table
(`tab:learned-policies`) compare the heuristic C2F and EG-C2F with two learned policies, Q-learning and REINFORCE.
The slide would show the same four curves, with F-IID dropped and each learned curve in the paper's color
(`policy_styles.json`, `learned`). Two standard objectives that both beat EG-C2F on the frozen observer (best
checkpoints) make the point about the observer rather than about one algorithm. This needs `<canvit-policy-curves>` to
take several added curves, each with its color, read from `learned_policies.json` in place of
`sources/learned-policy.json`. It depends on the camera-ready's open choice of objectives and checkpoints.

**The learned curve's color.** If the slide keeps Q-learning alone, `style="--canvit-policy: #9467bd"` on its chart
gives it the paper's Q-learning color, as the deck's rule for viewing policies asks. Teal is also the deck's color
for the policy as a component, which argues for keeping it; either way the choice belongs in `TALK.md`.

**Which checkpoint.** The deck shows each training run's best checkpoint, as the camera-ready's figure does. If the
paper prints the last checkpoints instead, `sources/learned-policy.json` takes the `selection: "last"` entry, so the
slide shows what the paper prints.

**"These results are unpublished."** Once the camera-ready keeps the appendix, the notes can say: "These results are
new; they'll be in the camera-ready version of the paper."

## Pretraining CanViT-B (`#pretraining`)

**Teacher features are optional to precompute.** A bullet after "Computing DINOv3's features once took about 8 more
GPU-hours", in the camera-ready's words (`subsec:distillation`): "Precomputation of teacher features is entirely
optional, and is a tradeoff between storage and compute cost." `supp:pretraining-details` estimates the total
H100-hours with the teacher computed online instead. `TALK.md` records that precomputing the teacher's features can
be divisive; the paper now answers the point.

## The paper slide (`#paper`)

**Lamarck Labs among the affiliations.** The camera-ready lists Lamarck Labs as an affiliation of both speakers, as
`#title` does; `#paper`'s logo row stops at Université Laval. The change is the `#title` markup
`<span class="wordlogo lamarck"><img src="assets/logos/lamarck-labs.svg" alt="">Lamarck Labs</span>` after Laval,
with a `#paper .logos .wordlogo` rule in ink. The row is already as wide as the four portraits, so the logos need to
shrink or the row to split; check it on a screenshot.

## Backup slides for questions the camera-ready now answers

Each would sit after `#frontier`, built from the export named, with its conditions in the notes.

- **"How much of this is DINOv3?"** The architecture control (`supp:architecture-control`, `architecture_control.json`):
  CanViT-B against an AdaGlimpse baseline with the same DINOv3 teacher, ImageNet-1k images and glimpse budget, mIoU by
  glimpse against cumulative GFLOPs, and the training chip-hours of each. It isolates the architecture under a fixed
  teacher; DINOv3's absolute contribution stays unmeasured, as `sources/checked-facts.md` says.
- **"Is it ImageNet-21k?"** Pretraining on ImageNet-1k alone (`supp:dataset-scale`, `dataset_scale.json`), still ahead
  of every model in the paper's comparison with prior work (`tab:active-vision-comparison`). The camera-ready has an
  open choice about which ImageNet-21k values to print.
- **"What else can the canvas do?"** COCO detection (`supp:coco-detection`, `coco_detection.json`): AP by glimpse for
  the flagship with a CenterNet head. The flagship's fine-tuning recipe differs from the architecture control's pair,
  so the slide shows the flagship alone or both with their recipes.
- **"What does the VPE token do?"** One bullet in `#architecture-paper`'s notes: its benefit at long horizons with a
  64 × 64 canvas on 1024 px scenes (`supp:vpe-long-horizon`).
