"""Export CanViT-B and its ADE20K probe for <canvit-live>, check the export, publish it (docs/viz.md, "Live model").

    uv run --extra live python -m canvit_pytorch.viz.live export --out-dir ../site/.live-model
    uv run --extra live python -m canvit_pytorch.viz.live parity --model-dir ../site/.live-model --image IMAGE
    uv run --extra live python -m canvit_pytorch.viz.live check-browser --page-url URL --reference-url URL \
        --backend webgpu --out REPORT
    uv run --extra live python -m canvit_pytorch.viz.live publish --model-dir ../site/.live-model --out-dir STAGING [--push]
"""

import logging

import tyro

from canvit_pytorch.viz.live.browser import CheckBrowser
from canvit_pytorch.viz.live.export import Export
from canvit_pytorch.viz.live.parity import Parity
from canvit_pytorch.viz.live.publish import Publish

log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    command = tyro.extras.subcommand_cli_from_dict({"export": Export, "parity": Parity, "check-browser": CheckBrowser, "publish": Publish})
    log.info(f"Wrote {command.run()}")
