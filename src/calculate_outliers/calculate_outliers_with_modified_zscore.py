import polars as pl

def calculate_outliers_with_modified_zscore(
    df: pl.DataFrame,
    value_col: str,
    group_col: str | None = None,
    threshold: float = 3.5
) -> pl.DataFrame:
    """Berechnet Modified Z-Scores und Ausreißer-Flags für eine numerische Spalte.

    Unterstützt eine globale Berechnung oder eine gruppierte Berechnung, wenn
    ``group_col`` gesetzt ist.

    Args:
        df: Eingabe-DataFrame als Polars DataFrame.
        value_col: Name der zu analysierenden numerischen Spalte.
        group_col: Optionale Gruppierungsspalte. Falls gesetzt, werden Median
            und MAD innerhalb jeder Gruppe berechnet.
        threshold: Absoluter Modified-Z-Score-Schwellenwert, ab dem ein Wert
            als Ausreißer markiert wird.

    Returns:
        Eingabe-DataFrame erweitert um:
        - ``mod_zscore``: Modified Z-Score pro Zeile.
        - ``is_outlier``: Boolescher Ausreißer-Indikator.
    """
    # Basismedian entweder global oder je Gruppe berechnen.
    median_expr = pl.col(value_col).median()
    
    if group_col:
        median_expr = median_expr.over(group_col)

    # Pipeline:
    # 1) Median je Zeile verfügbar machen
    # 2) MAD robust berechnen
    # 3) numerische Stabilisierung gegen sehr kleine MAD-Werte
    # 4) Modified Z-Score + Outlier-Flag erzeugen
    return (
        df.with_columns(
            _median = median_expr
        )
        .with_columns(
            # MAD muss den eben berechneten Median der jeweiligen Zeile nutzen
            _mad_raw = (pl.col(value_col) - pl.col("_median")).abs().median()
        )
        .with_columns(
            # Auch hier: Falls Gruppe vorhanden, MAD über Gruppe berechnen
            _mad_raw = pl.col("_mad_raw").over(group_col) if group_col else pl.col("_mad_raw")
        )
        .with_columns(
            _mad = pl.max_horizontal(pl.col("_mad_raw"), pl.col("_median").abs() * 0.01)
        )
        .with_columns(
            mod_zscore = (0.6745 * (pl.col(value_col) - pl.col("_median")) / pl.col("_mad"))
        )
        .with_columns(
            is_outlier = pl.col("mod_zscore").abs() > threshold
        )
        .drop("_median", "_mad_raw", "_mad")
    )
