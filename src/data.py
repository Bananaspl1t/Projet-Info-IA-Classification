"""Chargement du fichier Excel (onglet « Données »)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import build_features

TARGET = "categorie"


def load_dataset(path: str | Path, exclude_jan1: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Lit le fichier et retourne (X, y).

    Colonnes attendues : nom, date_naissance, pays, categorie.
    Le nom n'est jamais utilisé comme variable.
    """
    df = pd.read_excel(path, sheet_name="Données")
    missing = {"date_naissance", "pays", TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier : {sorted(missing)}")
    df = df.dropna(subset=["date_naissance", TARGET])
    if exclude_jan1:  # dates probablement saisies par défaut
        d = pd.to_datetime(df["date_naissance"])
        df = df[~((d.dt.month == 1) & (d.dt.day == 1))]
    X = build_features(df["date_naissance"], df["pays"])
    y = df[TARGET].reset_index(drop=True)
    return X, y
