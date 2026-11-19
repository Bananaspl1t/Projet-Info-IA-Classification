"""Pipelines scikit-learn et jeux de variables comparés."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42

# Jeux de variables comparés (ablation). Le dernier est le modèle du projet.
FEATURE_SETS = {
    "mois+jour_seul": ["mois", "jour"],
    "pays_seul": ["pays"],
    "pays+mois": ["pays", "mois"],
    "pays+mois+jour": ["pays", "mois", "jour"],
}
FULL = "pays+mois+jour"


def preprocessor(columns: list[str]) -> ColumnTransformer:
    # Toutes les variables sont catégorielles. Les modalités rares (< 10 personnes,
    # surtout des pays) sont regroupées ; les modalités inconnues aussi.
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), columns)]
    )


def make_model(kind: str, columns: list[str]) -> Pipeline:
    if kind == "majorite":
        clf = DummyClassifier(strategy="prior")
    elif kind == "aleatoire":
        clf = DummyClassifier(strategy="stratified", random_state=SEED)
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=3000)
    elif kind == "foret":
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("prep", preprocessor(columns)), ("clf", clf)])
