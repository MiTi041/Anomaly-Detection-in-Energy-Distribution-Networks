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
    """
    Bereinigt den DataFrame iterativ von Ausreißern.
    
    Args:
        df: Input DataFrame.
        cols_to_check: Liste der Spalten, die geprüft werden sollen.
        plot: Wenn True, werden die Plots während des Laufs angezeigt.
        threshold: Schwellenwert für das Outlier-Flag (Standard 3.5).
        
    Returns:
        tuple: (clean_df, report_dict)
    """
    clean_df = df.clone()
    report = {}

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
    """
    Wandelt Daten in das Format [Samples, TimeSteps, Features] um.
    """
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
    """
    Berechnet Anomalien in einem DataFrame mittels Anomaliedetection.

    Die Funktion markiert Ausreißer in den angegebenen Spalten und ergänzt den
    DataFrame um die Spalten `is_outlier` (bool). Zeilen mit fehlenden Werten in den Features werden als normal
    behandelt (`is_outlier=False`).

    Args:
        df (pl.DataFrame): Der Eingabe-DataFrame mit den zu analysierenden Spalten.
        value_cols (str | list[str]): Name einer Spalte oder Liste von Spalten,
            die für die Anomalie-Erkennung verwendet werden sollen.
        threshold (float | str, optional): Schwellenwert für den Rekonstruktionsfehler. 
            Bei 'auto' wird der maximale Fehler der Trainingsdaten verwendet. Defaults to 'auto'.
        window_size (int, optional): Anzahl der Zeitschritte pro Sequenz. 
            Muss für diese Architektur ein Vielfaches von 4 sein. Defaults to 16.
        cleaning_data_threshold (float, optional): Schwellenwert für die vorab 
            durchgeführte iterative Datenbereinigung mittels Modified Z-Score. 
            Werte, deren Z-Score diesen Wert überschreiten, werden aus dem 
            Trainingsdatensatz entfernt, um eine unverfälschte Normalverteilung 
            für das Autoencoder-Training zu gewährleisten. Höhere Werte bedeuten 
            eine tolerantere Bereinigung. Defaults to 3.5.

    Returns:
        pl.DataFrame: Ein Polars DataFrame, das die Originaldaten enthält, ergänzt um:
            - `is_outlier` (bool): True für vom Autoencoder erkannte Anomalien, 
              False für normale Punkte.

    Hinweise:
        - Nullwerte in den verwendeten Spalten werden für das Modell ignoriert,
          erhalten aber im Output `is_outlier=False`.
        - Skalierung mittels StandardScaler erfolgt automatisch, um unterschiedliche
          Wertebereiche kompatibel zu machen.
        - Die Funktion eignet sich sowohl für univariate als auch multivariate Daten.
        - Die Reihenfolge der Zeilen im Ausgangs-DataFrame bleibt erhalten.

    Beispiel:
    .. code-block:: python
        df_with_outliers = calculate_outliers_with_autoencoder(
            df=df,
            value_cols=["forecast_error", "percentage_forecast_error"],
            treshold=3
        )
        # Zugriff auf Anomalien:
        anomalies = df_with_outliers.filter(pl.col("is_outlier"))
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    if window_size % 4 != 0:
        raise ValueError("window_size muss für diese Modellarchitektur durch 4 teilbar sein.")

    df_clean, _ = clean_data(df, cols_to_check=value_cols, plot=True, threshold=cleaning_data_threshold)

    fig = plot_time_series(
        df=df_clean,
        use_log_scale=True,
        value_cols=["demand_mw_sum", "demand_forecast_mw_sum", "net_generation_mw_sum"],
        colors={"demand_mw_sum": "blue", "demand_forecast_mw_sum": "orange", "net_generation_mw_sum": "green"},
        title=f"Zeitreihe: Demand (MW) Summiert vs. Demand Forecast (MW) Summiert vs. Net Generation (MW) Summiert - Daten bereinigt - von {min(df_clean['utc_time_end']).strftime('%d.%m.%Y')} bis {max(df_clean['utc_time_end']).strftime('%d.%m.%Y')}"
    )

    fig.show()
    
    # Skalierung (Pflicht für Isolation Forest bei mehreren Spalten)
    X = df_clean.select(value_cols).to_numpy()

    # NOTE ALT:
    # scaler = StandardScaler()
    # oder:
    # scaler = RobustScaler()
    
    # NOTE NEU: PowerTransformer macht, dass positive und negative  gleich stark bestraft werden
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X) # Fit auf sauberen Daten!
    X_3d = create_sequences(X_scaled, window_size)
    
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

    fig.update_layout(
        width=900,
        height=450
    )

    fig.show()

    # Get train MAE loss.
    x_train_pred = model.predict(X_3d)
    train_mse_loss = np.mean(np.abs(x_train_pred - X_3d), axis=(1, 2))

    # NOTE ALT:
    # calculated_threshold = np.max(train_mse_loss)
    
    # NOTE NEU (Viel besser):
    # Wir nehmen das 99. Perzentil. Alles, was schlechter ist als 
    # 99% der Trainingsdaten, ist eine Anomalie.
    calculated_threshold = np.percentile(train_mse_loss, 99)
    print("Reconstruction error threshold: ", calculated_threshold)

    
    # NOTE Hier sieht man klar dass der berechnete Threshold fast ganz rechts in der Verteilung ist. Das ist schlecht wie müssen das Percentiel verricngern um einen besseren Threshold zu bekommen. !!!
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

    # NOTE Jetzt verringern wir den Percentiel-Wert (Optimal wäre, wenn der Threshold am Ende der Klippe ist und vor der Flaute)
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

    X_dirty = df.drop_nulls(subset=value_cols).select(value_cols).to_numpy()
    X_dirty_scaled = scaler.transform(X_dirty) 
    X_3d_dirty = create_sequences(X_dirty_scaled, window_size)

    error = np.mean(np.square(X_3d_dirty - model.predict(X_3d_dirty)), axis=(1, 2))
    
    # 2. Schneide den working_df passend zu
    # Wir nehmen die letzten n_predictions Zeilen
    valid_df = df.drop_nulls(subset=value_cols).slice(window_size - 1)
    
    # 3. Jetzt haben beide exakt die gleiche Länge (n_predictions)!
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