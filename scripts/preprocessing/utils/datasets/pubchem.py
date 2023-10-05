"""Utilities for querying PubCHEM drug properties."""

from __future__ import annotations

import requests
import time

import pandas as pd

from tqdm import tqdm


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

PUBCHEM_DEFAULT_PROPERTIES = [
    "CanonicalSMILES",
    "InChIKey",
    "MolecularFormula",
    "MolecularWeight",
    "Title",
]


def fetch_pubchem_properties(
    pubchem_cids: list[str], properties: list[str] | None = None
) -> pd.DataFrame:
    """Queries PubCHEM properties for the specified PubCHEM compound ids."""

    if properties is None:
        properties = PUBCHEM_DEFAULT_PROPERTIES

    properties_str = ",".join(properties)
    url_fmt = f"{PUBCHEM_BASE_URL}/{{}}/property/{properties_str}/JSON"

    query_results = []
    num_requests = 0

    for cid in tqdm(pubchem_cids, desc="Fetching PubCHEM properties"):
        try:
            resp = requests.get(url_fmt.format(cid))
            resp_json = resp.json()
            query_results.append(resp_json["PropertyTable"]["Properties"][0])
        except Exception as e:
            raise e

        num_requests += 1
        if num_requests % 5 == 0:
            time.sleep(0.5)  # avoid rate limiting errors

    return pd.DataFrame(query_results)
