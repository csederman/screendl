#!/usr/bin/env python
"""Downloads and saves the GDSC drug list."""

import click
import requests

import pandas as pd

from datetime import datetime
from io import StringIO
from pathlib import Path


url = "https://www.cancerrxgene.org/api/compounds"


@click.command()
@click.option("--output-dir", help="Output directory.")
def main(output_dir: str) -> None:
    """Fetch GDSCv2 compounds list."""
    output_dir = Path(output_dir)

    ts = datetime.now().strftime("%Y_%m_%d")

    with requests.Session() as s:
        resp = s.get(url, params={"list": "all", "export": "csv"})
        content = resp.content.decode()
        GDSC_drug_list = pd.read_csv(StringIO(content))

    GDSC_drug_list.columns = [c.strip() for c in GDSC_drug_list.columns]
    GDSC_drug_list.write_csv(output_dir / f"drug_list_{ts}.csv")


if __name__ == "__main__":
    main()
