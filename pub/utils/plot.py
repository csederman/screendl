"""Plotting utilities for ScreenDL publication."""

from __future__ import annotations

import altair as alt

from enum import Enum


class NPG5Palette(str, Enum):
    """Plotting colors.

    From: https://nanx.me/ggsci/index.html
    """

    BLUE = "#3C5488"
    CYAN = "#4DBBD5"
    GREEN = "#01A087"
    ORANGE = "#F49B7F"
    RED = "#E64B35"


class NPGPalette(str, Enum):
    """Expanded NPG palette.

    From: https://nanx.me/ggsci/index.html
    """

    BLUE = "#89D0E2"
    BROWN = "#A5917F"
    CYAN = "#6BBDAB"
    CYAN_LIGHT = "#B3DFD4"
    ORANGE = "#F08375"
    ORANGE_LIGHT = "#F7B9A7"
    PURPLE = "#8794B2"
    PURPLE_DARK = "#647598"
    PURPLE_LIGHT = "#A9B2CB"
    RED = "#EA635C"


MODEL_COLORS = {
    "DeepCDR": NPGPalette.BROWN.value,
    "DualGCN": NPGPalette.RED.value,
    "HiDRA": NPGPalette.BLUE.value,
    "Random Forest (C)": NPGPalette.ORANGE_LIGHT.value,
    "Random Forest (P)": NPGPalette.ORANGE.value,
    "Random Forest": NPGPalette.ORANGE_LIGHT.value,
    "Ridge (C)": NPGPalette.CYAN_LIGHT.value,
    "Ridge (P)": NPGPalette.CYAN.value,
    "Ridge": NPGPalette.CYAN.value,
    "ScreenDL-FT": NPGPalette.PURPLE.value,
    "ScreenDL-PT": NPGPalette.PURPLE_LIGHT.value,
    "ScreenDL-SA (ALL)": NPGPalette.PURPLE_DARK.value,
    "ScreenDL-SA (NBS)": NPGPalette.PURPLE_DARK.value,
    "ScreenDL-SA": NPGPalette.PURPLE_DARK.value,
}

MODEL_SHAPES = {
    "DeepCDR": "circle",
    "HiDRA": "circle",
    "Random Forest (C)": "circle",
    "Random Forest (P)": "diamond",
    "Random Forest": "circle",
    "Ridge (C)": "circle",
    "Ridge (P)": "diamond",
    "Ridge": "circle",
    "ScreenDL-FT": "triangle",
    "ScreenDL-PT": "square",
    "ScreenDL-SA (ALL)": "circle",
    "ScreenDL-SA (NBS)": "diamond",
    "ScreenDL-SA": "circle",
}


DEFAULT_BOXPLOT_CONFIG = {
    "box": alt.MarkConfig(stroke="black"),
    "median": alt.MarkConfig(fill="black"),
    "outliers": alt.MarkConfig(stroke="black", size=15, strokeWidth=1.5),
    "size": 30,
    "ticks": alt.MarkConfig(size=10),
}


DEFAULT_AXIS_CONFIG = {
    "domainColor": "black",
    "domainWidth": 1,
    "labelFont": "arial",
    "labelFontSize": 10,
    "tickColor": "black",
    "tickWidth": 1,
    "titleFont": "arial",
    "titleFontStyle": "regular",
    "titlePadding": 10,
}


def configure_chart(chart: alt.Chart) -> alt.Chart:
    """Configures boxplot for viewing."""
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(**DEFAULT_AXIS_CONFIG)
        .configure_header(
            labelFont="arial", titleFont="arial", titleFontStyle="regular"
        )
        .configure_legend(
            labelFont="arial", titleFont="arial", titleFontStyle="regular"
        )
    )
