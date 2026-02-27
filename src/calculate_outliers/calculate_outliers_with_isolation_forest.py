import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def calculate_outliers_with_isolation_forest(
    df: pl.DataFrame, 
    value_cols: str | list[str], 
    contamination: float | str = 'auto',
    random_state: int = 42
) -> pl.DataFrame:
    """
    Berechnet Anomalien in einem DataFrame mittels Isolation Forest.

    Die Funktion markiert Ausreißer in den angegebenen Spalten und ergänzt den
    DataFrame um die Spalten `is_outlier` (bool) und `if_score` (numerischer
    Anomalie-Score). Zeilen mit fehlenden Werten in den Features werden als normal
    behandelt (`is_outlier=False`).

    Args:
        df (pl.DataFrame): Der Eingabe-DataFrame mit den zu analysierenden Spalten.
        value_cols (str | list[str]): Name einer Spalte oder Liste von Spalten,
            die für die Anomalie-Erkennung verwendet werden sollen.
        contamination (float | str, optional): Erwarteter Anteil an Ausreißern. 
            Kann eine Zahl zwischen 0 und 1 sein oder 'auto'. Defaults to 'auto'.
        random_state (int, optional): Seed für die Reproduzierbarkeit der Ergebnisse. 
            Defaults to 42.

    Returns:
        pl.DataFrame: Ein Polars DataFrame, das die Originaldaten enthält, ergänzt um:
            - `is_outlier` (bool): True für vom Isolation Forest erkannte Anomalien, 
              False für normale Punkte.
            - `if_score` (float): Rohscore des Isolation Forest, höhere Werte = normaler,
              niedrigere Werte = anomal.

    Hinweise:
        - Nullwerte in den verwendeten Spalten werden für das Modell ignoriert,
          erhalten aber im Output `is_outlier=False`.
        - Skalierung mittels StandardScaler erfolgt automatisch, um unterschiedliche
          Wertebereiche kompatibel zu machen.
        - Die Funktion eignet sich sowohl für univariate als auch multivariate Daten.
        - Die Reihenfolge der Zeilen im Ausgangs-DataFrame bleibt erhalten.

    Beispiel:
    .. code-block:: python
        df_with_outliers = calculate_outliers_with_isolation_forest(
            df=df,
            value_cols=["forecast_error", "percentage_forecast_error"],
            contamination=0.01
        )
        # Zugriff auf Anomalien:
        anomalies = df_with_outliers.filter(pl.col("is_outlier"))
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    # Daten vorbereiten (Null-Werte müssen für sklearn raus)
    # Wir merken uns die ursprünglichen Indizes nicht, da Isolation Forest 
    # zeilenweise arbeitet. Wir nutzen drop_nulls auf den Features.
    working_df = df.drop_nulls(subset=value_cols)
    X = working_df.select(value_cols).to_numpy()
    
    # Skalierung (Pflicht für Isolation Forest bei mehreren Spalten)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modell-Fitting
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = clf.fit_predict(X_scaled)
    scores = clf.decision_function(X_scaled)
    
    # Ergebnis-Spalten zum working_df hinzufügen
    # Mapping: -1 (Anomalie) -> True, 1 (Normal) -> False
    res_df = working_df.with_columns([
        pl.Series("is_outlier", predictions == -1),
        pl.Series("if_score", scores)
    ])
    
    # Zurück-Joinen an den Original-DF, damit keine Zeilen verloren gehen
    # (Zeilen mit Nulls in value_cols erhalten null in den neuen Spalten)
    return df.join(
        res_df.select(list(df.columns) + ["is_outlier", "if_score"]),
        on=df.columns,
        how="left"
    ).with_columns(pl.col("is_outlier").fill_null(False))