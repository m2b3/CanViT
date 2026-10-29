"""How long each slide took in a recorded run-through, for the speaker notes (deck.js shows each slide's range of the
run's clock beside its budget). A slide starts at the first word of the phrase that opened it in the run; the first
slide starts with the take, and the last ends with the take's last word. Reads a transcription's result.json (its
utterances' word timestamps) and writes sources/rehearsal-<run>.json.

    uv run python -m experiments.rehearsal_timing --result PATH/result.json --manifest PATH/manifest.json \\
        --run 2026-10-27-1956 --take-start "super excited" --take-end "happy to take questions"   # from the deck
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import tyro

# The phrase each slide opened with in the run-through of 2026-10-27 at 19:56, in slide order (the title slide's is the
# take's start). A later run whose wording differs needs its own list.
OPENINGS = [
    ("title", None), ("tradition", "deep artificial neural networks have"), ("foundation", "a notable example"),
    ("dinov3-and-the-brain", "was at the center"), ("passive-models", "these models are great"),
    ("human-vision", "our vision is inherently"), ("timeline", "drawing inspiration from"),
    ("gap", "although active vision models"), ("active-vision-model", "to better understand this gap"),
    ("task-dependent-scanpaths", "we just talked about"), ("compare", "that was a lot"), ("paper", "thank you for this great"),
    ("canvit-rollout", "set up the problem"), ("outline-architecture", "to actually build this model"),
    ("vit-on-a-glimpse", "go back to what we would do"), ("integrating-information-across-viewpoints", "theres a problem with this"),
    ("architecture", "that motivates why"), ("sr-rope", "a key component of this process"),
    ("canvas-attention", "to zoom in a bit more"), ("outline-pretraining", "how we actually trained"),
    ("how-to-see-first", "one of the key ideas"), ("pretraining-policies", "during training we presented"),
    ("distillation", "what do we do with those"), ("pretraining", "some numbers on pretraining"),
    ("table-callback", "to go back to the example"), ("outline-experiments", "now lets get to experiments"),
    ("state-of-the-art", "semantic segmentation canvit sets"), ("c2f-and-f2c", "to zoom in on how we evaluated"),
    ("eg-c2f", "another policy that is"), ("policy-accuracy", "we looked at the accuracy"),
    ("frontier", "what about computational efficiency"), ("learned-policy", "heuristic policies are great"),
    ("quickstart", "so canvaset is open source"), ("next", "whats next"), ("links", "thats it for me"),
]


@dataclass
class Args:
    result: Path
    manifest: Path
    run: str
    take_start: str
    take_end: str
    out_dir: Path = Path("sources")


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def main(args: Args) -> None:
    utterances = json.loads(args.result.read_text())["result"]["transcription"]["utterances"]
    words = [(w["start"], w["end"], norm(w["word"])) for u in utterances for w in u["words"]]
    words = [w for w in words if w[2]]
    tokens = [w[2] for w in words]

    def find(phrase: str, after: int, last: bool = False) -> int:
        p = phrase.split()
        hits = [i for i in range(after, len(tokens) - len(p) + 1) if tokens[i:i + len(p)] == p]
        if not hits:
            raise SystemExit(f"rehearsal_timing: {phrase!r} not found in the run after word {after}")
        return hits[-1] if last else hits[0]

    take = find(" ".join(norm(w) for w in args.take_start.split()), 0, last=True)  # the last take
    end = find(" ".join(norm(w) for w in args.take_end.split()), take)
    end += len(args.take_end.split()) - 1
    starts, cursor = [], take
    for slide, phrase in OPENINGS:
        index = take if phrase is None else find(phrase, cursor)
        starts.append((slide, words[index][0]))
        cursor = index + 1
    bounds = starts + [("end", words[end][1])]
    seconds = {slide: round(b - a, 1) for (slide, a), (_, b) in zip(bounds, bounds[1:])}
    manifest = json.loads(args.manifest.read_text())
    out = args.out_dir / f"rehearsal-{args.run}.json"
    out.write_text(json.dumps({
        "_about": "Each slide's time in a recorded run-through, measured from its transcription's word timestamps by "
                  "experiments/rehearsal_timing.py: a slide starts at the first word of the phrase it opened with.",
        "run": args.run,
        "recording_sha256": manifest["audio"]["sha256"],
        "total_seconds": round(words[end][1] - words[take][0], 1),
        "seconds": seconds,
    }, indent=1) + "\n")
    print(f"wrote {out}: {len(seconds)} slides, {sum(seconds.values()):.0f} s")


if __name__ == "__main__":
    main(tyro.cli(Args))
