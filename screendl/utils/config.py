"""Config parsing utils."""

from __future__ import annotations

import typing as t

from omegaconf.listconfig import ListConfig


def safe_lconfig_as_tuple(item: t.Any) -> t.Any:
    """Converts ListConfig instances to tuples or does nothing."""
    return tuple(item) if isinstance(item, ListConfig) else item