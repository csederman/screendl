"""Utilities for querying PubCHEM drug properties."""

from __future__ import annotations

import json
import requests
import time

import pandas as pd
import typing as t

from pathlib import Path
from tqdm import tqdm


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

PUBCHEM_DEFAULT_PROPERTIES = [
    "CanonicalSMILES",
    "InChIKey",
    "MolecularFormula",
    "MolecularWeight",
    "Title",
]


def _load_cached_properties(fs_cache: Path) -> t.Dict[str, t.Any]:
    """Loads cached PubCHEM properties."""
    cached_props = dict()
    if fs_cache.exists():
        with open(fs_cache, "r") as fh:
            cached_props.update(json.load(fh))
    return cached_props


def fetch_pubchem_properties(
    pubchem_cids: t.List[str], fs_cache: t.Optional[str | Path]
) -> pd.DataFrame:
    """Queries PubCHEM properties for the specified PubCHEM compound ids.

    Parameters
    ----------
        pubchem_cids: A list of PubCHEM compound IDs to query.
        fs_cache: Path where queried properties should be stored.

    Returns
    -------
        A `pd.DataFrame` of PubCHEM properties
    """
    cached_props = _load_cached_properties(Path(fs_cache))

    properties_str = ",".join(PUBCHEM_DEFAULT_PROPERTIES)
    url_fmt = f"{PUBCHEM_BASE_URL}/{{}}/property/{properties_str}/JSON"

    query_results = []
    num_requests = 0
    for cid in tqdm(pubchem_cids, desc="Fetching PubCHEM properties"):
        if cid in cached_props:
            query_results.append(cached_props[cid])
            continue

        # fetch the properties from PubCHEM
        try:
            resp = requests.get(url_fmt.format(cid))
            resp_json = resp.json()
            props = resp_json["PropertyTable"]["Properties"][0]

            query_results.append(props)
            cached_props[cid] = props

        except Exception as e:
            raise e

        num_requests += 1
        if num_requests % 5 == 0:
            time.sleep(0.5)  # avoid rate limiting errors

    with open(fs_cache, "w", encoding="utf-8") as fh:
        json.dump(cached_props, fh, ensure_ascii=False, indent=4)

    return pd.DataFrame(query_results)
