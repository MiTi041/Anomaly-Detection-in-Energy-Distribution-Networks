from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import warnings

def plot_multivariate_data(
    df: pl.DataFrame,
    value_cols: list[str],
    group_col: str | None = None,
    sample_fraction: float | None = None,
    outlier_col: str | None = None,
    title: Optional[str] = None,
    columns: int = 1,
) -> go.Figure:
    """
    Universelle Funktion zur Visualisierung multivariater Daten mit optionaler Anomalie-Hervorhebung
    und optionaler Gruppierung.

    Erstellt 2D- oder 3D-Scatterplots für die angegebenen numerischen Spalten. 
    Optional können Punkte als Anomalien markiert und farblich hervorgehoben werden.
    Bei Angabe einer Gruppenspalte wird für jede Gruppe ein eigener Subplot erstellt.
    Achsen werden standardmäßig logarithmisch skaliert, um große Wertebereiche handhabbar darzustellen.

    Args:
        df (pl.DataFrame): Der Eingabe-Datensatz als Polars DataFrame.
        value_cols (list[str]): Liste von 2 oder 3 numerischen Spalten, die geplottet werden sollen.
        group_col (str, optional): Spalte für die Gruppierung der Subplots. 
            Defaults to None (alle Punkte in einem Plot).
        sample_fraction (float, optional): Anteil der Stichprobe für große Datenmengen (0 < sample_fraction < 1).
            Defaults to None (ganzer Datensatz).
        outlier_col (str, optional): Name einer booleschen Spalte, die Anomalien markiert.
            True = Anomalie, False = normal. Defaults to None.
        title: Optional[str] = None,
        columns (int, optional): Anzahl der Spalten auf die sich die Sub-Plots verteilen.
            Defaults to 1.

    Returns:
        go.Figure: Ein Plotly Figure-Objekt mit den generierten Scatterplots.
            - 2D-Scatter bei 2 Spalten, 3D-Scatter bei 3 Spalten.
            - Punkte werden nach Normalität oder Anomalie farblich unterschieden.
            - Optional Subplots pro Gruppe.
            - Achsen logarithmisch, Titel enthält Hinweis auf Anomalieerkennung und log-Skalierung.

    Hinweise:
        - Bei Gruppierung entsteht pro Gruppe ein Subplot.
        - Sampling dient nur der Performance und verändert nicht die Anomaliekennzeichnung.
        - Logarithmische Achsen komprimieren kleine Werte; Null- oder Negativwerte müssen ggf. vorher behandelt werden.

    Beispiel:
    .. code-block:: python
        fig = plot_multivariate_data(
            df=df,
            value_cols=["forecast_error", "percentage_forecast_error"],
            group_col="region",
            sample_fraction=0.1,
            outlier_col="is_anomaly"
        )
        fig.show()
    """

    if len(value_cols) != 2:
        raise ValueError("Nur 2 werden momentan unterstützt.")

    colors = {"normal": "blue", "not_outlier": "lightgray", "outlier": "red"}
    plot_df = df.drop_nulls(subset=value_cols)

    if sample_fraction is not None and 0 < sample_fraction < 1:
        plot_df = plot_df.sample(fraction=sample_fraction, seed=42)

    # --- Gruppierung vorbereiten ---
    if group_col is None:
        plot_df = plot_df.with_columns(pl.lit("Gesamt").alias("temp_group"))
        current_group_col = "temp_group"
    else:
        current_group_col = group_col

    groups = plot_df[current_group_col].unique().to_list()
    n_groups = len(groups)

    # Dynamische Layout-Parameter
    n_rows = (n_groups + columns - 1) // columns
    dynamic_height = 150 * n_rows
    if n_groups == 1:
        dynamic_height = 400  # z.B. höhere Höhe für nur einen Plot
    else:
        dynamic_height = max(300, 150 * n_rows)
    dynamic_spacing = max(0.01, min(0.1, 1 / (n_rows * 2))) if n_rows > 1 else 0.1

    subplot_titles = []
    for group_val in groups:
        df_sub = plot_df.filter(pl.col(current_group_col) == group_val)
        total = df_sub.height
        if outlier_col and outlier_col in df_sub.columns:
            n_outliers = df_sub.filter(pl.col(outlier_col)).height
            subplot_titles.append(f"{group_val} (<b>{n_outliers}</b> von <b>{total}</b> Punkten sind Anomalien)")
        else:
            subplot_titles.append(f"{group_val}")
        
    fig = make_subplots(
        rows=n_rows, 
        cols=columns, 
        subplot_titles=subplot_titles,
        vertical_spacing=dynamic_spacing,
        shared_xaxes=False
    )

    def add_trace(data, label, color, size, opacity, row, col, showlegend=True):
        if data.height == 0:
            return
        
        fig.add_trace(go.Scattergl(
            x=data[value_cols[0]],
            y=data[value_cols[1]],
            mode="markers",
            name=label,
            showlegend=showlegend,
            marker=dict(color=color, size=size, opacity=opacity),
            hovertemplate="<br>".join([
                f"{value_cols[0]}: %{{x}}",
                f"{value_cols[1]}: %{{y}}"
            ]) + "<extra></extra>"
        ), row=row, col=col)

    title_suffix = ""

    if outlier_col:
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode="markers", name="Normal",
            marker=dict(color=colors["not_outlier"], size=5), showlegend=True
        ))
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode="markers", name="Anomalie",
            marker=dict(color=colors["outlier"], size=6), showlegend=True
        ))

    for i, group_val in enumerate(groups):
        row = i // columns + 1
        col = i % columns + 1
        df_sub = plot_df.filter(pl.col(current_group_col) == group_val)

        if outlier_col and outlier_col in df_sub.columns:
            df_not_outlier = df_sub.filter(~pl.col(outlier_col))
            df_outlier = df_sub.filter(pl.col(outlier_col))
            add_trace(df_not_outlier, "Normal", colors["not_outlier"], 4, 0.4, row=row, col=col, showlegend=False)
            add_trace(df_outlier, "Anomalie", colors["outlier"], 6, 0.9, row=row, col=col, showlegend=False)
            title_suffix = " - Anomalieerkennung"
            if group_col:
                title_suffix += f" - Gruppiert nach {group_col}"
        else:
            if outlier_col and outlier_col not in df_sub.columns:
                warnings.warn(f"Spalte '{outlier_col}' existiert nicht im DataFrame. Anomalien werden ignoriert.")
                
            add_trace(df_sub, "Normal", colors["normal"], 5, 0.5, row=row, col=col, showlegend=True)

    # Layout
    title_text = f"Multivariater Scatter: {' vs. '.join(value_cols)}{title_suffix} (log-Skala)"
    fig.update_layout(
        height=dynamic_height,
        width=1200 if n_groups > 1 else 900,
        title=title or title_text,
        legend_title="Status" if outlier_col else None
    )

    if len(value_cols) == 3:
        fig.update_layout(scene=dict(
            xaxis_title=value_cols[0],
            yaxis_title=value_cols[1],
            zaxis_title=value_cols[2],
        ))

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    fig.update_annotations(font_size=10)

    return fig