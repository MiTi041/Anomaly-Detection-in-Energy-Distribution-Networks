import polars as pl

def calculate_outliers_with_modified_zscore(
    df: pl.DataFrame,
    value_col: str,
    group_col: str | None = None,
    threshold: float = 3.5
) -> pl.DataFrame:
    """Berechnet den Modified Z-Score (global oder gruppiert).
    
    Args:
        df: Polars DataFrame.
        value_col: Die zu prüfende Metrik (z.B. 'demand_mw').
        group_col: Optional; Spalte für die Gruppierung. Wenn None, wird global berechnet.
        threshold: Schwellenwert für das Outlier-Flag (Standard 3.5).
        
    Returns:
        pl.DataFrame: Der originale DataFrame ergänzt um 'mod_zscore' und 'is_outlier'.
    """
    
    median_expr = pl.col(value_col).median()
    
    if group_col:
        median_expr = median_expr.over(group_col)

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