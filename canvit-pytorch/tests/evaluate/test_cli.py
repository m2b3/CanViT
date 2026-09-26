import pytest
import tyro

from canvit_pytorch.evaluate.__main__ import COMMANDS


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_help_works_without_datasets(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building every command's defaults for --help must not resolve $ADE20K_ROOT or $IMAGENET_VAL."""
    monkeypatch.delenv("ADE20K_ROOT", raising=False)
    monkeypatch.delenv("IMAGENET_VAL", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        tyro.extras.subcommand_cli_from_dict(COMMANDS, args=[command, "--help"])
    assert exit_info.value.code == 0
