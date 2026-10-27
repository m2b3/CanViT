import io
import logging
import sys

import pytest

from canvit_pytorch.progress import progress


def test_progress_logs_the_start_and_the_end_when_stderr_is_not_a_terminal(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    with caplog.at_level(logging.INFO, logger="canvit_pytorch.progress"):
        assert list(progress(range(3), description="Steps")) == [0, 1, 2]
    assert len(caplog.messages) == 2, caplog.messages
    assert caplog.messages[0].startswith("Steps:   0%") and "0/3" in caplog.messages[0]
    assert caplog.messages[1].startswith("Steps: 100%") and "3/3" in caplog.messages[1]
    assert not stderr.getvalue(), "a log carries no carriage-return bar"
