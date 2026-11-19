"""Prédiction : catégorie la plus probable pour une date de naissance et un pays.

Usage : python -m src.predict --date 1990-05-12 --country Russie
"""
from __future__ import annotations

import argparse
from datetime import date

import joblib
import pandas as pd

from .features import build_features

DISCLAIMER = (
    "Projet pédagogique : ces probabilités reflètent surtout la composition de la liste "
    "(pays émetteurs, fréquences). Aucun lien causal. Une notice rouge n'est pas une condamnation. "
    "Ne pas utiliser sur de vraies personnes."
)


def predict(model_path: str, birth: date, country: str, top: int = 3) -> list[tuple[str, float]]:
    bundle = joblib.load(model_path)
    X = build_features(pd.Series([pd.Timestamp(birth)]), pd.Series([country]))[bundle["columns"]]
    proba = bundle["model"].predict_proba(X)[0]
    ranked = sorted(zip(bundle["classes"], proba), key=lambda t: -t[1])
    return ranked[:top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="AAAA-MM-JJ")
    ap.add_argument("--country", required=True, help="Nom du pays en français, ex. Russie")
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    birth = date.fromisoformat(args.date)
    print(f"Naissance : {birth.day:02d}/{birth.month:02d} | Pays : {args.country}")
    for cat, p in predict(args.model, birth, args.country, args.top):
        print(f"  {p:6.1%}  {cat}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
