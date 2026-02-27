from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

# Zentrale Datenpfade für alle Loader-Funktionen.
BALANCE_DATA_PATH = Path("./data/balance.parquet")
INTERCHANGE_DATA_PATH = Path("./data/interchange.parquet")
SUBREGION_DATA_PATH = Path("./data/subregion.parquet")

# Einheitliches Plotly-Template für reproduzierbare Visualisierungen.
pio.templates["scientific"] = go.layout.Template(
    layout=go.Layout(
        width=900, 
        height=450,
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#111111"
        ),
        title=dict(
            font=dict(size=16, color="#111111"),
            x=0.5,  # zentriert
            xanchor='center',
        ),
        coloraxis=dict(
            colorscale="Viridis",  # angenehme, perceptually uniform
            colorbar=dict(
                title=dict(font=dict(size=12)),
                tickfont=dict(size=10)
            )
        ),
        legend=dict(
            title_font=dict(size=12),
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#e0e0e0",
            borderwidth=1,
            itemsizing="constant"
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=False,
            showline=True,
            linecolor="#111111",
            ticks="outside",
            ticklen=5,
            automargin=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=False,
            showline=True,
            linecolor="#111111",
            ticks="outside",
            ticklen=5,
            automargin=True
        ),
    )
)

pio.templates.default = "scientific"
