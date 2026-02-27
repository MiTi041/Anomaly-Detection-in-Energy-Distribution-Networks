# Anomaly Detection in Energy Distribution Networks

Dieses Repository enthält die Implementierung eines robusten Systems zur Anomalieerkennung in Energieverteilungsnetzwerken. Im Rahmen der akademischen Ausarbeitung wurden Ansätze aus dem Bereich Deep Learning (Autoencoder) und klassischem Machine Learning (Isolation Forest) evaluiert und implementiert.

## 🚀 Kernmerkmale

- **Autoencoder-Architektur:** Implementierung mittels `Keras`/`TensorFlow` unter Verwendung von `Conv1D` und `Conv1DTranspose` Layern.
- **Methodik:** Einsatz von **GELU-Aktivierungsfunktionen** und **PowerTransformer (Yeo-Johnson)** zur Normalisierung rechtsschiefer Energiedaten.
- **Mathematische Konsistenz:** Anomalie-Detektion basiert auf einer mathematisch konsistenten Schwellenwertanalyse (Mean Squared Error - MSE).
- **Methodenvergleich:** Integration eines `IsolationForest` als Baseline-Verfahren zur Validierung der Ergebnisse.
- **Datenverarbeitung:** Hochperformante Datenmanipulation mittels `Polars`.

## 🛠 Tech Stack

- **Sprache:** Python 3.10+
- **Data Handling:** `Polars`
- **Deep Learning:** `Keras`, `TensorFlow`
- **ML-Library:** `Scikit-Learn`
- **Visualisierung:** `Plotly`

## 📥 Installation

1. Repository klonen:

   ```bash
   git clone [https://github.com/DEIN_USERNAME/DEIN_REPO_NAME.git](https://github.com/DEIN_USERNAME/DEIN_REPO_NAME.git)
   cd DEIN_REPO_NAME

   ```

2. Virtuelle Umgebung erstellen und Abhängigkeiten installieren:
   python -m venv venv

   # Windows:

   venv\Scripts\activate

   # Linux/macOS:

   source venv/bin/activate

   # Abhängigkeiten installieren

   pip install -r requirements.txt

## 📊 Verwendung

Das System ist modular aufgebaut. Die Anomalieerkennung kann direkt auf einen Polars-DataFrame angewendet werden:

```python
import polars as pl
from src.anomaly_detection import calculate_outliers_with_autoencoder

# Beispiel für die Anwendung
df = pl.read_csv("data/energy_grid_data.csv")

# Anomalieerkennung mit dem Autoencoder
df_results = calculate_outliers_with_autoencoder(
    df=df,
    value_cols=["demand_mw_sum", "net_generation_mw_sum"],
    window_size=16,
    threshold='auto'
)

# Ergebnis filtern
anomalies = df_results.filter(pl.col("is_outlier") == True)
print(f"Anzahl erkannter Anomalien: {len(anomalies)}")
```

## 📂 Repository Struktur

├── data/ # (Ignoriert) Rohdaten
├── src/ # Quellcode
│ ├── plotting.py # Visualisierungs-Logik
│ └── anomaly_detection.py # Autoencoder & Logging Logik
├── notebooks/ # Explorative Analyse (EDA)
├── requirements.in # Top-Level Abhängigkeiten
├── requirements.txt # Generierte Abhängigkeiten (lock-file)
└── README.md # Projektdokumentation

## 📝 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Informationen finden Sie in der [LICENSE](LICENSE) Datei.
