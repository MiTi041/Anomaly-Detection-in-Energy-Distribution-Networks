# Anomaly Detection in Energy Distribution Networks

## Kurzfassung

Im Rahmen dieser Arbeit wird die Erkennung von Anomalien in Zeitreihen elektrischer Energieverteilungsnetze untersucht. Grundlage sind stündliche Balancing-Authority-Daten (Demand, Forecast, Net Generation, Interchange). Ziel ist der methodische Vergleich eines robusten statistischen Verfahrens (`Modified Z-Score`), eines klassischen Machine-Learning-Verfahrens (`Isolation Forest`) und eines rekonstruktionsbasierten Deep-Learning-Ansatzes (`Autoencoder`).

## Problemstellung und Zielsetzung

Energieverteilungsnetze weisen durch Lastschwankungen, Prognosefehler, Messartefakte und operative Ereignisse komplexe, nichtstationäre Muster auf. Klassische Schwellwertregeln sind in diesem Kontext oft unzureichend.

Die Arbeit verfolgt folgende Zielsetzung:

- Entwicklung einer reproduzierbaren Pipeline zur Datenaufbereitung und Anomalieerkennung
- Vergleich mehrerer Verfahren hinsichtlich Robustheit und praktischer Eignung
- Analyse der Erkennbarkeit atypischer Last- und Erzeugungsverläufe

## Forschungsfragen

1. In welchem Umfang unterscheiden sich die Ergebnisse statistischer, baumbasierter und neuronaler Verfahren auf denselben Zeitreihen?
2. Welche Vorverarbeitungsschritte sind für stabile und nachvollziehbare Ergebnisse erforderlich?
3. Welche methodischen Grenzen ergeben sich ohne vollständig gelabelte Anomaliedaten?

## Datengrundlage

Verwendet werden Parquet-Dateien im Verzeichnis `data/`:

- `balance.parquet`
- `interchange.parquet`
- `subregion.parquet`

Die standardmäßigen Dateipfade sind in `src/config.py` definiert. Das Laden, Umbenennen und Typisieren der Rohdaten erfolgt in `src/data_loader.py`.

## Methodik

### 1. Datenaufbereitung

- Harmonisierung von Spaltennamen und Datumsformaten
- Filterung unvollständiger bzw. nicht geeigneter Zeiträume
- Feature-Bildung (u. a. `forecast_error`, `percentage_forecast_error`)
- Konsistenzprüfungen und Duplikatkontrolle

### 2. Anomalieerkennung

- `Modified Z-Score`: robuste baselinebasierte Einzelsignal-Detektion
- `Isolation Forest`: multivariate, nichtlineare Outlier-Erkennung
- `Autoencoder` (Conv1D): Rekonstruktionsfehler-basiertes Verfahren auf Sequenzen

### 3. Auswertung

- Kennzeichnung verdächtiger Zeitpunkte über `is_outlier`
- Visuelle Plausibilisierung mit Plotly-Zeitreihen und Verteilungen
- Vergleich der Verfahren hinsichtlich Sensitivität und Stabilität

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

## Reproduzierbarkeit

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

### Empfohlene Ausführungsreihenfolge

1. `datenanalyse_balance.ipynb`
2. `datenanalyse_interchange.ipynb`
3. `datenanalyse_subregion.ipynb`
4. `isolation_forest.ipynb`
5. `autoencoder.ipynb`

## Methodische Einschränkungen

- Fehlende Ground-Truth-Labels erschweren eine eindeutige quantitative Bewertung.
- Schwellenwerte (z. B. beim Rekonstruktionsfehler) beeinflussen die Detektionsrate stark.
- Zeitfensterwahl und Feature-Selektion wirken sich direkt auf die Ergebnisstabilität aus.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht. Details siehe [LICENCE](LICENCE).
