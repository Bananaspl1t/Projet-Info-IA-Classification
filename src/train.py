"""Entraînement, comparaison avec des références et test de permutation.

Usage : python -m src.train --data data/liste_rouge_categorisee.xlsx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, permutation_test_score

from .data import load_dataset
from .models import FEATURE_SETS, FULL, SEED, make_model

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "log_loss": "neg_log_loss",
}


def evaluate(kind: str, columns: list[str], X, y, cv) -> dict:
    res = cross_validate(make_model(kind, columns), X[columns], y, cv=cv, scoring=SCORING, n_jobs=-1)
    out = {m: float(np.mean(res[f"test_{m}"])) for m in SCORING}
    out["log_loss"] = -out["log_loss"]  # perte positive : plus petit = mieux
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/liste_rouge_categorisee.xlsx")
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--exclude-jan1", action="store_true", help="Exclure les naissances au 1er janvier (dates par défaut probables)")
    ap.add_argument("--out-model", default="models/model.joblib")
    ap.add_argument("--out-report", default="reports/metrics.json")
    args = ap.parse_args()

    X, y = load_dataset(args.data, exclude_jan1=args.exclude_jan1)
    print(f"{len(X)} personnes, {y.nunique()} catégories")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    # 1) Références : toujours répondre la catégorie majoritaire / tirage selon les fréquences
    for kind in ("majorite", "aleatoire"):
        rows.append({"modele": kind, "variables": "-", **evaluate(kind, FEATURE_SETS[FULL], X, y, cv)})
    # 2) Modèles réels, avec ablation des variables
    for name, cols in FEATURE_SETS.items():
        for kind in ("logreg", "foret"):
            rows.append({"modele": kind, "variables": name, **evaluate(kind, cols, X, y, cv)})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))

    # 3) Test de permutation : le mois et le jour apportent-ils quelque chose en plus du pays ?
    base_cols = FEATURE_SETS["pays_seul"]
    full_cols = FEATURE_SETS[FULL]
    perm = {}
    for label, cols in (("pays_seul", base_cols), (FULL, full_cols)):
        score, _, pval = permutation_test_score(
            make_model("logreg", cols), X[cols], y, cv=cv, scoring="neg_log_loss",
            n_permutations=args.permutations, random_state=SEED, n_jobs=-1,
        )
        perm[label] = {"log_loss": float(-score), "p_value": float(pval)}
        print(f"Permutation [{label}] log-loss={-score:.4f}  p={pval:.4f}")

    # 4) Modèle final : meilleure perte logarithmique avec toutes les variables
    full = table[table["variables"] == FULL].sort_values("log_loss")
    best_kind = full.iloc[0]["modele"]
    final = make_model(best_kind, full_cols).fit(X[full_cols], y)
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "columns": full_cols, "classes": list(final.classes_)}, args.out_model)

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    report = {"n": int(len(X)), "final_model": best_kind, "cv_table": table.to_dict("records"), "permutation": perm}
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modèle final : {best_kind} -> {args.out_model}")


if __name__ == "__main__":
    main()
