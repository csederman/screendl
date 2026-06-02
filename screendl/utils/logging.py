"""Logging utilities."""

from __future__ import annotations

import logging
import sys

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """Console logging handler that does not corrupt tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_tqdm_logging(
    *,
    level: int | str | None = None,
    replace_stream_handlers: bool = True,
) -> None:
    """Route console logs through tqdm.write.

    This preserves file handlers, including Hydra's file logging, but replaces
    stdout/stderr stream handlers so console logs do not break tqdm bars.
    """
    root = logging.getLogger()

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    tqdm_handler = TqdmLoggingHandler(stream=sys.stderr)
    tqdm_handler.setFormatter(formatter)

    if level is not None:
        tqdm_handler.setLevel(level)
        root.setLevel(level)

    if replace_stream_handlers:
        for handler in list(root.handlers):
            is_stream = isinstance(handler, logging.StreamHandler)
            is_file = isinstance(handler, logging.FileHandler)

            # FileHandler is a StreamHandler subclass, so preserve it explicitly.
            if is_stream and not is_file:
                root.removeHandler(handler)

    # Avoid adding duplicate tqdm handlers if called repeatedly by Hydra multirun.
    if not any(isinstance(h, TqdmLoggingHandler) for h in root.handlers):
        root.addHandler(tqdm_handler)