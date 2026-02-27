import plotly.graph_objects as go
import polars as pl
from typing import Optional, List, Dict
import warnings

def plot_time_series(
    df: pl.DataFrame,
    value_cols: List[str],
    colors: Optional[Dict[str, str]] = None,
    outlier_col: Optional[str] = None,
    title: Optional[str] = None,
    use_log_scale: bool = False,
    marker_size: float = 15
) -> go.Figure:
    """
    Zeitreihenplot mit logarithmischer Y-Achse und optionaler Anomalie-Hervorhebung.

    Args:
        df (pl.DataFrame): Polars DataFrame mit Zeitstempel 'utc_time_end'.
        value_cols (List[str]): Numerische Spalten zum Plotten.
        colors (Dict[str,str], optional): Mapping Spaltenname -> Farbe.
        outlier_col (str, optional): Boolesche Spalte zur Markierung von Anomalien.
        title (str, optional): Eigener Plot-Titel.
        use_log_scale (bool, optional): Flag, ob die Achsen Log-Skaliert werden sollen. Defaults to False.

    Returns:
        go.Figure: Plotly-Figur.
    """

    if "utc_time_end" not in df.columns:
        raise ValueError("DataFrame muss die Spalte 'utc_time_end' enthalten.")

    fig = go.Figure()

    default_colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    if colors is None:
        colors = {col: default_colors[i % len(default_colors)] for i, col in enumerate(value_cols)}

    for col in value_cols:
        if col not in df.columns:
            continue

        line_color = colors.get(col, "blue")

        # Basislinie
        fig.add_trace(go.Scatter(
            x=df["utc_time_end"].to_list(),
            y=df[col].to_list(),
            mode="lines",
            name=col,
            line=dict(color=line_color),
            opacity=0.5
        ))

        # Ausreißer markieren
        if outlier_col and outlier_col in df.columns:
            mask = df[outlier_col].to_list()
            outlier_y = [val if is_outlier else None for val, is_outlier in zip(df[col].to_list(), mask)]
            fig.add_trace(go.Scatter(
                x=df["utc_time_end"].to_list(),
                y=outlier_y,
                mode="markers",
                marker=dict(size=marker_size, color="red"),
            ))
            fig.add_trace(go.Scatter(
                x=df["utc_time_end"].to_list(),
                y=outlier_y,
                mode="lines",
                name=f"{col} Outlier",
                line=dict(color="red"),
            ))
        
        if outlier_col and outlier_col not in df.columns:
                warnings.warn(f"Spalte '{outlier_col}' existiert nicht im DataFrame. Anomalien werden ignoriert.")

    fig.update_layout(
        title=title or f"Zeitreihe: {' vs. '.join(value_cols)}",
        xaxis_title="UTC Time",
        yaxis_title="Wert (log-skaliert)" if use_log_scale else "Wert",
        yaxis_type="log" if use_log_scale else None,
        legend_title="Spalten",
        hovermode="x unified",
        xaxis=dict(
            showspikes=True,          # vertikale Linie aktivieren
            spikecolor="rgba(100,100,100,0.5)",  # halbtransparente graue Linie
            spikethickness=1,         # etwas dickere Linie
            spikedash="solid",         # durchgezogen statt gepunktet
            spikemode="across",       # Linie über alle Linien
            spikesnap="cursor",       # Linie folgt genau dem Maus-Cursor
            showline=True,            # x-Achsenlinie anzeigen           # schöne Lesbarkeit der Zeitachse
        ),
        yaxis=dict(
            showspikes=True,
            spikecolor="rgba(100,100,100,0.5)",
            spikethickness=1,
            spikedash="solid",
            spikesnap="cursor",
            gridcolor="lightgray",    # harmonische Rasterlinien
            gridwidth=0.5,
        ),
        hoverlabel=dict(
            bgcolor="white",      # Hintergrundfarbe des Blocks
            bordercolor="gray",   # Umrandung
            font_size=12,
            font_family="Arial",
        )
    )

    return fig