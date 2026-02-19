#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from collections import Counter
import pandas as pd


def norm(s):
    """Normalizza stringhe (lower, strip, spazi)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def freq_weighted_overlap(base: Counter, other: Counter) -> float:
    """
    Quota dei token (conteggi) del base che hanno lemma presente anche in other:
    sum_{w in base and w in other} base[w] / sum_{w in base} base[w]
    """
    denom = sum(base.values())
    if denom == 0:
        return 0.0
    shared_mass = sum(cnt for w, cnt in base.items() if w in other)
    return shared_mass / denom


def get_subset(df, type_name):
    """
    Estrae subset distractor per tipo.
    - type_name: 'pnpn', 'verbal', 'nsungiu'
    """
    # robust matching su colonna Type
    type_col = "Type"
    if type_col not in df.columns:
        raise ValueError(f"Colonna '{type_col}' non trovata. Colonne disponibili: {list(df.columns)}")

    t = df[type_col].astype(str).map(norm)

    if type_name == "pnpn":
        mask = t.str.contains(r"\bpnpn\b", regex=True)
    elif type_name == "verbal":
        mask = t.str.contains(r"\bverbal\b", regex=True)
    elif type_name == "nsungiu":
        # gestisce: nsungiu / nsungiù / nsungiù / nsungiù ecc.
        mask = t.str.contains(r"nsung", regex=True)  # sufficientemente robusto
    else:
        raise ValueError("type_name must be one of: pnpn, verbal, nsungiu")

    # prendiamo SOLO distractor (construction == no/false/0)
    c = df["construction"].astype(str).map(norm)
    distr_mask = c.isin({"no", "false", "0", "distr", "distractor"})
    return df[mask & distr_mask].copy()


def main():
    ap = argparse.ArgumentParser(
        description="Compute lexical overlap (noun lemmas) between CXN and distractor types (PNPN, VERBAL, NsuNgiù)."
    )
    ap.add_argument("--input", required=True, help="Path to the input CSV/TSV with ';' separator.")
    ap.add_argument("--sep", default=";", help="Field separator (default ';').")
    ap.add_argument("--noun_col", default="noun", help="Column containing noun lemmas (default 'noun').")
    ap.add_argument("--outdir", default="overlap_outputs", help="Output directory for CSV reports.")
    ap.add_argument("--topk", type=int, default=50, help="How many shared lemmas to print/save (sorted by CXN freq).")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=args.sep, dtype=str, keep_default_na=False)

    # Basic checks
    required_cols = {"construction", "Type", args.noun_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    os.makedirs(args.outdir, exist_ok=True)

    # Define CXN subset (construction == yes/true/1)
    c = df["construction"].astype(str).map(norm)
    cxn_df = df[c.isin({"yes", "true", "1", "cxn", "construction"})].copy()

    # Lemma lists / counters
    cxn_lemmas = cxn_df[args.noun_col].map(norm)
    cxn_lemmas = cxn_lemmas[cxn_lemmas != ""]
    cxn_ctr = Counter(cxn_lemmas.tolist())
    cxn_types = set(cxn_ctr.keys())

    print(f"\nLoaded: {args.input}")
    print(f"Rows total: {len(df)}")
    print(f"CXN rows: {len(cxn_df)} | CXN noun tokens: {sum(cxn_ctr.values())} | CXN noun types: {len(cxn_types)}")

    distractor_map = {
        "PNPN": "pnpn",
        "VERBAL": "verbal",
        "NSUNGIU": "nsungiu",
    }

    summary_rows = []

    for label, key in distractor_map.items():
        sub = get_subset(df, key)

        lemmas = sub[args.noun_col].map(norm)
        lemmas = lemmas[lemmas != ""]
        ctr = Counter(lemmas.tolist())
        types = set(ctr.keys())

        inter = cxn_types & types
        jac = jaccard(cxn_types, types)
        fwo = freq_weighted_overlap(cxn_ctr, ctr)

        print("\n" + "=" * 72)
        print(f"{label}")
        print(f"Rows: {len(sub)} | noun tokens: {sum(ctr.values())} | noun types: {len(types)}")
        print(f"Overlap types |CXN ∩ {label}|: {len(inter)}")
        print(f"Jaccard(types): {jac:.4f}")
        print(f"Freq-weighted overlap (CXN token mass shared): {fwo:.4f}")

        # Build shared list sorted by CXN frequency
        shared = [(w, cxn_ctr[w], ctr[w]) for w in inter]
        shared.sort(key=lambda x: x[1], reverse=True)
        shared_top = shared[: args.topk]

        # Print top shared
        print(f"\nTop shared lemmas (lemma, CXN_count, {label}_count) [top {args.topk}]:")
        for w, c1, c2 in shared_top:
            print(f"  {w}\t{c1}\t{c2}")

        # Save shared table
        out_csv = os.path.join(args.outdir, f"shared_lemmas_CXN_{label}.csv")
        pd.DataFrame(shared, columns=["lemma", "cxn_count", f"{label.lower()}_count"]).to_csv(out_csv, index=False)

        summary_rows.append({
            "distractor": label,
            "cxn_types": len(cxn_types),
            "distr_types": len(types),
            "overlap_types": len(inter),
            "jaccard_types": jac,
            "cxn_token_mass_shared": fwo,
        })

    # Save summary
    summary_path = os.path.join(args.outdir, "overlap_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nSaved outputs to:", args.outdir)
    print(" - overlap_summary.csv")
    print(" - shared_lemmas_CXN_PNPN.csv / shared_lemmas_CXN_VERBAL.csv / shared_lemmas_CXN_NSUNGIU.csv\n")


if __name__ == "__main__":
    main()
