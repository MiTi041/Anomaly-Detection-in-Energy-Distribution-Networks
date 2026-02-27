from typing import Optional
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import polars as pl
import warnings

def plot_univariate_data(
    df: pl.DataFrame,
    value_col: str, 
    group_col: str | None = None,
    outlier_col: str | None = None,
    show_density: bool = True,
    sample_fraction: float | None = None,
    title: Optional[str] = None,
    columns: int = 1,
) -> go.Figure:
    """Visualisiert univariate Verteilungen mit optionaler Gruppierung und Ausreißern.

    Jede Gruppe wird in einem eigenen Subplot mit Scatter-Punkten und optionaler
    Violin-Dichtedarstellung gezeigt.

    Args:
        df: Eingabe-DataFrame.
        value_col: Numerische Spalte, die visualisiert werden soll.
        group_col: Optionale Gruppierungsspalte für separate Subplots.
        outlier_col: Optionale boolesche Ausreißer-Flag-Spalte.
        show_density: Gibt an, ob Violin-Dichtedarstellungen gezeichnet werden.
        sample_fraction: Optionales Sampling-Verhältnis für Nicht-Ausreißer bei
            großen Datensätzen.
        title: Optionaler benutzerdefinierter Figure-Titel.
        columns: Anzahl der Subplot-Spalten.

    Returns:
        Plotly-Figure mit einem Subplot pro Gruppe (oder einem Gesamtplot).
    """
    # Einheitliche Farbsemantik für normale Punkte und Ausreißer.
    colors = {"normal": "blue", "not_outlier": "green", "outlier": "red"}
    
    # 1. Handling für optionale Gruppierung
    # Wenn keine Gruppe definiert ist, erstellen wir eine temporäre Konstante
    if group_col is None:
        plot_df = df.with_columns(pl.lit("Gesamt").alias("temp_group"))
        current_group_col = "temp_group"
    else:
        plot_df = df
        current_group_col = group_col

    # Gruppen bestimmen; jede Gruppe wird in einem eigenen Subplot gezeigt.
    groups = sorted(plot_df[current_group_col].unique().to_list())
    n_groups = len(groups)

    # 2. Dynamische Layout-Parameter
    n_rows = (n_groups + columns - 1) // columns
    dynamic_height = max(200, 120 * n_rows)
    dynamic_spacing = max(0.01, min(0.1, 1 / (n_rows * 2))) if n_rows > 1 else 0.01
    dynamic_violin_width = 1.5 if n_groups > 10 else 0.8

    # Subplot-Titel optional um Ausreißerzählung erweitern.
    subplot_titles = []
    for group_val in groups:
        df_sub = plot_df.filter(pl.col(current_group_col) == group_val)
        total = df_sub.height
        if outlier_col and outlier_col in df_sub.columns:
            n_outliers = df_sub.filter(pl.col(outlier_col)).height
            subplot_titles.append(f"{group_val} (<b>{n_outliers}</b> von <b>{total}</b> Punkten sind Anomalien)")
        else:
            subplot_titles.append(f"{group_val}")

    # Optionales Sampling: alle Ausreißer behalten, normale Punkte ausdünnen.
    if sample_fraction is not None and 0 < sample_fraction < 1 and outlier_col in plot_df.columns:
        df_outliers = plot_df.filter(pl.col(outlier_col))         # Alle Anomalien behalten
        df_normals = plot_df.filter(~pl.col(outlier_col))         # Nur normale Punkte
        df_normals_sampled = df_normals.sample(fraction=sample_fraction, seed=42)
        plot_df = pl.concat([df_outliers, df_normals_sampled])   # wieder zusammenführen
        
    fig = make_subplots(
        rows=n_rows, 
        cols=columns, 
        subplot_titles=subplot_titles,
        vertical_spacing=dynamic_spacing,
        shared_xaxes=False
    )

    # Dummy-Traces sorgen für eine saubere Legende über alle Subplots hinweg.
    if outlier_col:
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode="markers", name="Normal",
            marker=dict(color=colors["not_outlier"], size=5), showlegend=True
        ))
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode="markers", name="Anomalie",
            marker=dict(color=colors["outlier"], size=6), showlegend=True
        ))

    def add_trace(data, label, color, size, opacity, row, col, showlegend=True):
        """Fügt einem Ziel-Subplot eine Scatter-Trace für eine Teilmenge hinzu."""
        fig.add_trace(go.Scattergl(
            x=data[value_col],
            y=np.zeros(data.height),
            mode="markers",
            name=label,
            showlegend=showlegend,
            marker=dict(color=color, size=size, opacity=opacity),
            hovertemplate=f"{value_col}: %{{x}}" + "<extra></extra>"
        ), row=row, col=col)

    for i, group_val in enumerate(groups):
        row = i // columns + 1
        col = i % columns + 1
        df_sub = plot_df.filter(pl.col(current_group_col) == group_val)

        # Dichte-Komponente (Violine) für den Verteilungsüberblick.
        if show_density:
            fig.add_trace(go.Violin(
                x=df_sub[value_col],
                y0=0,
                orientation='h',
                side='positive',
                line_color='rgba(150, 150, 150, 0.3)',
                fillcolor='rgba(200, 200, 200, 0.15)',
                points=False,
                showlegend=False,
                width=dynamic_violin_width,
            ), row=row, col=col)
        
        # Punktdarstellung: entweder getrennt nach Outlier-Status oder einfarbig.
        if outlier_col and outlier_col in df_sub.columns:
            df_not_outlier = df_sub.filter(~pl.col(outlier_col))
            df_outlier = df_sub.filter(pl.col(outlier_col))
                
            if df_not_outlier.height > 0:
                add_trace(df_not_outlier, "Normal", colors["not_outlier"], 4, 0.4, row=row, col=col, showlegend=False)
            if df_outlier.height > 0:
                add_trace(df_outlier, "Anomalie", colors["outlier"], 6, 0.9, row=row, col=col, showlegend=False)
        
        # Wenn keine Score-Spalte -> Einfache einfarbige Verteilung
        else:
            if outlier_col and outlier_col not in df_sub.columns:
                warnings.warn(f"Spalte '{outlier_col}' existiert nicht im DataFrame. Anomalien werden ignoriert.")

            fig.add_trace(go.Scattergl(
                x=df_sub[value_col],
                y=np.zeros(df_sub.height),
                mode='markers',
                marker=dict(color=colors["normal"], size=6, opacity=0.4),
                showlegend=False, hoverinfo='x'
            ), row=row, col=col)

    # Layout abschließen: y-Achse dient nur als Jitter-/Platzhalterachse.
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor='black', 
                     showticklabels=False, range=[-0.2, 1.2], fixedrange=True)

    title_text = f"Verteilung: {value_col}"
    if group_col: title_text += f" - Gruppiert nach {group_col}"
    if outlier_col: title_text += f" - Anomalienerkennung"
    
    fig.update_layout(
        height=dynamic_height,
        width=1200 if n_groups > 1 else 900,
        title_text=title or title_text,
        legend_title="Status" if outlier_col else None
    )

    fig.update_annotations(font_size=10)
    
    return fig
