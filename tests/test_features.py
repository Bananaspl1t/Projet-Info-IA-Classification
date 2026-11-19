import pandas as pd

from src.features import build_features


def test_build_features_month_day_country():
    X = build_features(pd.Series(["1985-07-23"]), pd.Series(["Russie"]))
    assert list(X.columns) == ["pays", "mois", "jour"]
    assert (X.loc[0, "pays"], X.loc[0, "mois"], X.loc[0, "jour"]) == ("Russie", "7", "23")


def test_year_is_not_a_feature():
    X = build_features(pd.Series(["1985-07-23", "2001-07-23"]), pd.Series(["Russie", "Russie"]))
    assert X.loc[0].equals(X.loc[1])


def test_missing_country():
    X = build_features(pd.Series(["1985-07-23"]), pd.Series([None]))
    assert X.loc[0, "pays"] == "Inconnu"
