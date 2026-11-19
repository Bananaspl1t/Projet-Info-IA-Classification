"""Construction des variables : pays, mois et jour de naissance (sans année, sans signe)."""
from __future__ import annotations

import pandas as pd


def build_features(dates: pd.Series, countries: pd.Series) -> pd.DataFrame:
    """Transforme dates de naissance et pays en tableau de variables.

    Le mois et le jour sont traités comme des catégories (et non des nombres) :
    le 31 n'est pas « plus grand » que le 1er au sens d'un effet linéaire.
    """
    dates = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "pays": countries.fillna("Inconnu").astype(str).values,
            "mois": dates.dt.month.astype(str).values,
            "jour": dates.dt.day.astype(str).values,
        }
    )
