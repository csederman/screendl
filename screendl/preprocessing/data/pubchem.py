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


def _load_cached_properties(cache_path: Path) -> t.Dict[str, t.Any]:
    """Loads cached PubCHEM properties."""
    cached_props = dict()
    if cache_path.exists():
        with open(cache_path, "r") as fh:
            cached_props.update(json.load(fh))
    return cached_props


def fetch_pubchem_properties(
    pubchem_cids: t.List[str], cache_path: t.Optional[str | Path] = None
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
    properties_str = ",".join(PUBCHEM_DEFAULT_PROPERTIES)
    url_fmt = f"{PUBCHEM_BASE_URL}/{{}}/property/{properties_str}/JSON"

    query_props = {}
    cached_props = {}
    if isinstance(cache_path, (str, Path)):
        cached_props = _load_cached_properties(Path(cache_path))

    n_requests = 0
    for cid in tqdm(pubchem_cids, desc="Fetching PubCHEM properties"):
        cid = str(cid)
        if cid in cached_props:
            query_props[cid] = cached_props[cid]
            continue

        try:
            resp = requests.get(url_fmt.format(cid))
            resp.raise_for_status()
            query_props[cid] = resp.json()["PropertyTable"]["Properties"][0]

        except Exception as e:
            pass

        n_requests += 1
        if n_requests % 5 == 0:
            time.sleep(0.5)  # avoid rate limiting errors from PubCHEM

    if isinstance(cache_path, (str, Path)):
        cached_props.update(query_props)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cached_props, fh, ensure_ascii=False, indent=4)

    return pd.DataFrame.from_dict(query_props, orient="index")
