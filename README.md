# Anomaly Detection in Energy Distribution Networks

Python-Projekt zur Anomalieerkennung in Zeitreihendaten von Energieverteilungsnetzen.
Der Code enthält Datenaufbereitung, mehrere Outlier-Verfahren und Plot-Funktionen.

## Daten

Verwendet werden Parquet-Dateien im Verzeichnis `data/`:

- `balance.parquet`
- `interchange.parquet`
- `subregion.parquet`

Die Standardpfade sind in `src/config.py` definiert.

## Repository-Struktur

```text
.
├── data/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── calculate_outliers/
│   │   ├── calculate_outliers_with_modified_zscore.py
│   │   ├── calculate_outliers_with_isolation_forest.py
│   │   └── calculate_outliers_with_autoencoder.py
│   └── plotting/
├── datenanalyse_balance.ipynb
├── datenanalyse_interchange.ipynb
├── datenanalyse_subregion.ipynb
├── isolation_forest.ipynb
├── autoencoder.ipynb
└── requirements.txt
```

## Kernmodule

- `src/data_loader.py`:
  Lädt und normalisiert die Parquet-Daten, erzeugt Features und prüft auf Duplikate.
- `src/calculate_outliers/calculate_outliers_with_modified_zscore.py`:
  Robuste univariate Outlier-Erkennung mit Modified Z-Score.
- `src/calculate_outliers/calculate_outliers_with_isolation_forest.py`:
  Multivariate Outlier-Erkennung mit `sklearn` Isolation Forest.
- `src/calculate_outliers/calculate_outliers_with_autoencoder.py`:
  Sequenzbasierte Outlier-Erkennung mit Conv1D-Autoencoder (Keras).
- `src/plotting/`:
  Plotly-Funktionen für Zeitreihen-, univariate und multivariate Visualisierung.

## Setup

### Voraussetzungen

- Python `>= 3.10`
- Abhängigkeiten gemäß `requirements.txt`

### Installation

```bash
git clone <REPO-URL>
cd "Anomaly Detection in Energy Distribution Networks"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Nutzung

Der typische Ablauf im Code ist:

1. Daten laden (`get_balance_data`, `get_interchange_data`, `get_subregion_data`)
2. Outlier berechnen (Modified Z-Score, Isolation Forest oder Autoencoder)
3. Ergebnisse mit `is_outlier` weiterverarbeiten oder plotten

Beispiel (Isolation Forest):

```python
from src.data_loader import get_balance_data
from src.calculate_outliers.calculate_outliers_with_isolation_forest import (
    calculate_outliers_with_isolation_forest,
)

df, _ = get_balance_data()
result = calculate_outliers_with_isolation_forest(
    df=df,
    value_cols=["forecast_error", "percentage_forecast_error"],
    contamination=0.01,
)
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht. Details siehe [LICENCE](LICENCE).
