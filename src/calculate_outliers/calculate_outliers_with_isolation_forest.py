import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def calculate_outliers_with_isolation_forest(
    df: pl.DataFrame, 
    value_cols: str | list[str], 
    contamination: float | str = 'auto',
    random_state: int = 42
) -> pl.DataFrame:
    """Erkennt Ausreißer mit einem Isolation-Forest-Modell.

    Das Modell wird auf Zeilen ohne Nullwerte in ``value_cols`` trainiert.
    Vorhersagen und Scores werden anschließend an den vollständigen
    Eingabe-DataFrame zurückgeführt.

    Args:
        df: Eingabe-DataFrame.
        value_cols: Eine Feature-Spalte oder eine Liste von Feature-Spalten.
        contamination: Erwarteter Ausreißeranteil in den Daten oder ``"auto"``.
        random_state: Zufalls-Seed für reproduzierbares Modellverhalten.

    Returns:
        Eingabe-DataFrame erweitert um:
        - ``is_outlier``: ``True`` für als anomal erkannte Punkte.
        - ``if_score``: Entscheidungs-Score des Isolation Forest.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    # 1) Daten vorbereiten: sklearn akzeptiert keine Nullwerte.
    working_df = df.drop_nulls(subset=value_cols)
    X = working_df.select(value_cols).to_numpy()
    
    # 2) Skalierung für stabile Distanz-/Trennungsstruktur.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3) Modell trainieren und pro Zeile Vorhersage + Score ermitteln.
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = clf.fit_predict(X_scaled)
    scores = clf.decision_function(X_scaled)
    
    # 4) Vorhersagen in boolesches Outlier-Flag überführen.
    res_df = working_df.with_columns([
        pl.Series("is_outlier", predictions == -1),
        pl.Series("if_score", scores)
    ])
    
    # 5) Auf den Original-DF zurückjoinen, damit keine Zeilen verloren gehen.
    # Zeilen mit Nulls in den Features werden als nicht anomal markiert.
    return df.join(
        res_df.select(list(df.columns) + ["is_outlier", "if_score"]),
        on=df.columns,
        how="left"
    ).with_columns(pl.col("is_outlier").fill_null(False))
