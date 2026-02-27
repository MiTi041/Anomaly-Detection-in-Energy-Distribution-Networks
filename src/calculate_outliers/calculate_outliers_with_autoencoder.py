import polars as pl
import numpy as np
import keras
from keras import layers
from sklearn.preprocessing import PowerTransformer
import plotly.graph_objects as go
from src.plotting.plotting_time_series import plot_time_series
from src.plotting.plotting_univariate_data import plot_univariate_data
from src.calculate_outliers.calculate_outliers_with_modified_zscore import calculate_outliers_with_modified_zscore


def clean_data(
    df: pl.DataFrame, 
    cols_to_check: list[str], 
    plot: bool = True,
    threshold: float = 3.5
) -> tuple[pl.DataFrame, dict]:
    """Entfernt Ausreißer iterativ aus ausgewählten Spalten.

    Für jede Spalte werden Ausreißer per Modified Z-Score erkannt und entfernt,
    bevor die nächste Spalte verarbeitet wird.

    Args:
        df: Eingabe-DataFrame.
        cols_to_check: Numerische Spalten, die auf Ausreißer geprüft werden.
        plot: Gibt an, ob Zwischenplots angezeigt werden.
        threshold: Modified-Z-Score-Schwellenwert für die Ausreißererkennung.

    Returns:
        Tupel bestehend aus:
        - Bereinigtem DataFrame nach iterativer Filterung.
        - Report-Dictionary mit Anzahl entfernter Zeilen pro Spalte.
    """
    # Wir arbeiten auf einer Kopie, damit der Eingabe-DataFrame unverändert bleibt.
    clean_df = df.clone()
    report = {}

    # Jede Spalte wird nacheinander bereinigt; spätere Schritte profitieren von
    # den bereits gefilterten Zeilen.
    for col in cols_to_check:
        rows_before = len(clean_df)
        
        # 1. Anomalien berechnen
        temp_df = calculate_outliers_with_modified_zscore(clean_df, value_col=col, threshold=threshold)
        
        # 2. Visualisieren
        if plot:
            fig = plot_univariate_data(
                temp_df, 
                outlier_col="is_outlier", 
                value_col=col, 
                sample_fraction=0.2
            )
            fig.show()
        
        # 3. Filtern und Hilfsspalte löschen
        clean_df = temp_df.filter(pl.col("is_outlier") == False).drop("is_outlier")
        
        rows_after = len(clean_df)
        report[col] = rows_before - rows_after
        
        print(f"Spalte '{col}': {rows_before - rows_after} Ausreißer entfernt. Verbleibende Zeilen: {rows_after}")

    return clean_df, report

def create_sequences(data, window_size):
    """Erzeugt überlappende Zeitfenster für Sequenzmodelle.

    Args:
        data: Zweidimensionale Eingabe mit Form ``(n_samples, n_features)``.
        window_size: Anzahl Zeitschritte pro Sequenz.

    Returns:
        NumPy-Array mit Form ``(n_windows, window_size, n_features)``.
    """
    # Sliding-Window-Erzeugung: jedes Fenster wird als eigenes Sample genutzt.
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)

def calculate_outliers_with_autoencoder(
    df: pl.DataFrame, 
    value_cols: str | list[str], 
    threshold: float | str = 'auto',
    window_size: int = 16,
    cleaning_data_threshold: float = 3.5
) -> pl.DataFrame:
    """Erkennt Ausreißer in Zeitreihen mit einem konvolutionalen Autoencoder.

    Die Funktion bereinigt zunächst Trainingsdaten per Modified Z-Score,
    trainiert dann einen Autoencoder auf gleitenden Fenstern und markiert
    Punkte mit hohem Rekonstruktionsfehler als Ausreißer.

    Args:
        df: Eingabe-DataFrame.
        value_cols: Feature-Spalte oder Liste von Feature-Spalten.
        threshold: Gewünschte Schwellenwert-Strategie. Wird aktuell aus
            Kompatibilitätsgründen akzeptiert; die Schwelle wird in dieser
            Implementierung aus einem Perzentil des Trainingsfehlers abgeleitet.
        window_size: Sequenzlänge für den Modelleingang. Muss für die
            Encoder/Decoder-Architektur durch 4 teilbar sein.
        cleaning_data_threshold: Modified-Z-Score-Schwellenwert für die
            Vorbereinigung vor dem Modelltraining.

    Returns:
        DataFrame mit Autoencoder-basierten Ausreißerlabels in ``is_outlier``.

    Raises:
        ValueError: Falls ``window_size`` nicht durch 4 teilbar ist.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    if window_size % 4 != 0:
        raise ValueError("window_size muss für diese Modellarchitektur durch 4 teilbar sein.")

    # 1) Trainingsdaten vorab robust bereinigen, damit das Modell "Normalzustand"
    # lernt und nicht bereits vorhandene Ausreißer rekonstruiert.
    df_clean, _ = clean_data(df, cols_to_check=value_cols, plot=True, threshold=cleaning_data_threshold)

    # 2) Bereinigte Referenzzeitreihe zur visuellen Plausibilisierung anzeigen.
    fig = plot_time_series(
        df=df_clean,
        use_log_scale=True,
        value_cols=["demand_mw_sum", "demand_forecast_mw_sum", "net_generation_mw_sum"],
        colors={"demand_mw_sum": "blue", "demand_forecast_mw_sum": "orange", "net_generation_mw_sum": "green"},
        title=f"Zeitreihe: Demand (MW) Summiert vs. Demand Forecast (MW) Summiert vs. Net Generation (MW) Summiert - Daten bereinigt - von {min(df_clean['utc_time_end']).strftime('%d.%m.%Y')} bis {max(df_clean['utc_time_end']).strftime('%d.%m.%Y')}"
    )

    fig.show()
    
    # 3) Feature-Matrix extrahieren und stabil transformieren.
    X = df_clean.select(value_cols).to_numpy()
    
    # PowerTransformer (Yeo-Johnson) stabilisiert Schiefe und ist auch für
    # nicht-positive Werte geeignet.
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X)  # Fit erfolgt nur auf bereinigten Daten.
    X_3d = create_sequences(X_scaled, window_size)
    
    # 4) Sequenz-Autoencoder: Encoder komprimiert, Decoder rekonstruiert.
    model = keras.Sequential(
        [
            layers.Input(shape=(X_3d.shape[1], X_3d.shape[2])),
            layers.Conv1D(
                filters=32,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="gelu",
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="gelu",
            ),
            layers.Conv1DTranspose(
                filters=16,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="gelu",
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="gelu",
            ),
            layers.Conv1DTranspose(filters=X_3d.shape[2], kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    # 5) Rekonstruktionslernen mit Early Stopping zur Vermeidung von Overfitting.
    history = model.fit(
        X_3d,
        X_3d,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=history.history["loss"],
        mode="lines",
        name="Training Loss",
        line=dict(color="blue"),),
    )

    fig.add_trace(go.Scattergl(
        x=history.history["val_loss"],
        mode="lines",
        name="Validation Loss",
        line=dict(color="orange"),),
    )

    fig.show()

    # 6) Rekonstruktionsfehler auf Trainingsfenstern als Referenzverteilung.
    x_train_pred = model.predict(X_3d)
    train_mse_loss = np.mean(np.abs(x_train_pred - X_3d), axis=(1, 2))

    # Zuerst konservative Schwelle (99. Perzentil) für visuelle Einordnung.
    calculated_threshold = np.percentile(train_mse_loss, 99)
    print("Reconstruction error threshold: ", calculated_threshold)

    # Histogramm zur Beurteilung, wo die Schwelle relativ zur Fehlerverteilung liegt.
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=train_mse_loss,
        nbinsx=50,
        marker_color='#4c78a8',
        opacity=0.7,
        name='Reconstruction Error'
    ))

    # Schwellenwert als vertikale Linie hinzufügen
    fig.add_vline(
        x=calculated_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Threshold ({calculated_threshold:.4f})",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Verteilung der Rekonstruktionsfehler",
        xaxis_title="MSE (Mean Squared Error)",
        yaxis_title="Häufigkeit",
        bargap=0.1, # Kleiner Abstand zwischen den Balken für bessere Lesbarkeit
    )

    fig.show()

    # Für höhere Sensitivität wird die operative Schwelle auf 95. Perzentil gesetzt.
    calculated_threshold = np.percentile(train_mse_loss, 95)
    print("Reconstruction error threshold: ", calculated_threshold)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=train_mse_loss,
        nbinsx=50,
        marker_color='#4c78a8',
        opacity=0.7,
        name='Reconstruction Error'
    ))

    # Schwellenwert als vertikale Linie hinzufügen
    fig.add_vline(
        x=calculated_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Threshold ({calculated_threshold:.4f})",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Verteilung der Rekonstruktionsfehler",
        xaxis_title="MSE (Mean Squared Error)",
        yaxis_title="Häufigkeit",
        bargap=0.1, # Kleiner Abstand zwischen den Balken für bessere Lesbarkeit
        width=900,
        height=450
    )

    fig.show()

    # 7) Modell auf die unbereinigten Originaldaten anwenden.
    X_dirty = df.drop_nulls(subset=value_cols).select(value_cols).to_numpy()
    X_dirty_scaled = scaler.transform(X_dirty) 
    X_3d_dirty = create_sequences(X_dirty_scaled, window_size)

    # Fehler pro Fenster als Anomalie-Score.
    error = np.mean(np.square(X_3d_dirty - model.predict(X_3d_dirty)), axis=(1, 2))
    
    # Fenster liefern erst ab Position (window_size - 1) gültige Vorhersagen.
    valid_df = df.drop_nulls(subset=value_cols).slice(window_size - 1)
    
    # Score-basierte Entscheidung: Auto-Schwelle oder explizit gesetzter Wert.
    res_df = valid_df.with_columns([
        pl.Series("is_outlier", error > (calculated_threshold if threshold == 'auto' else threshold))
    ])
    
    # Zurück-Joinen an den Original-DF, damit keine Zeilen verloren gehen
    # (Zeilen mit Nulls in value_cols erhalten null in den neuen Spalten)
    return df.join(
        res_df.select(list(df.columns) + ["is_outlier"]),
        on=df.columns,
        how="left"
    ).with_columns(pl.col("is_outlier").fill_null(False))
