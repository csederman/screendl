#!/usr/bin/env python
"""ScreenDL data preprocessing."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import typing as t

from . import benchmark as bmk

file_path = os.path.dirname(os.path.realpath(__file__))
initialize_params = bmk.make_initialize_params(file_path)


def preprocess(g_parameters: t.Dict[str, t.Any]) -> None:
    """"""
