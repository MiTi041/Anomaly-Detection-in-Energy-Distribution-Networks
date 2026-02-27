import polars as pl
from .config import BALANCE_DATA_PATH, INTERCHANGE_DATA_PATH, SUBREGION_DATA_PATH

def get_balance_data():
    """Lädt, normalisiert und validiert Daten auf Balancing-Authority-Ebene.

    Die Funktion liest die Balance-Parquet-Quelle, harmonisiert Spaltennamen,
    parst Datums-/Zeitfelder, filtert den relevanten Zeitraum, berechnet
    Forecast-Metriken, prüft auf Duplikate und gibt sortierte Daten zurück.

    Returns:
        Tupel bestehend aus:
        - Aufbereitetem Balance-DataFrame.
        - Liste fuel-bezogener Spaltennamen für Konsistenzprüfungen.

    Raises:
        ValueError: Falls doppelte Datensätze erkannt werden.
    """
    # Für Explorationszwecke keine Begrenzung der Tabellenzeilen in Polars.
    pl.Config.set_tbl_rows(-1)
    
    # 1) Rohdaten laden.
    df = pl.read_parquet(BALANCE_DATA_PATH)

    # Alle Spalten rausfiltern, die _imputed oder _adjusted enthalten
    cols_to_keep = [c for c in df.columns if not (c.endswith("(Imputed)") or c.endswith("(Adjusted)"))]
    df = df.select(cols_to_keep)

    # 2) Spaltennamen vereinheitlichen (snake_case + stabile Schlüssel).
    df = df.rename({
        "Balancing Authority": "balancing_authority",
        "Data Date": "date",
        "Hour Number": "hour",
        "Local Time at End of Hour": "local_time_end",
        "UTC Time at End of Hour": "utc_time_end",

        "Demand Forecast (MW)": "demand_forecast_mw",
        "Demand (MW)": "demand_mw",
        "Net Generation (MW)": "net_generation_mw",
        "Total Interchange (MW)": "total_interchange_mw",
        "Sum(Valid DIBAs) (MW)": "sum_valid_dibas_mw",

        # --- Fuel types ---
        "Net Generation (MW) from Coal": "coal_mw",
        "Net Generation (MW) from Natural Gas": "natural_gas_mw",
        "Net Generation (MW) from Nuclear": "nuclear_mw",
        "Net Generation (MW) from All Petroleum Products": "petroleum_mw",
        "Net Generation (MW) from Hydropower Excluding Pumped Storage": "hydro_mw",
        "Net Generation (MW) from Pumped Storage": "pumped_storage_mw",
        "Net Generation (MW) from Solar without Integrated Battery Storage": "solar_mw",
        "Net Generation (MW) from Solar with Integrated Battery Storage": "solar_battery_mw",
        "Net Generation (MW) from Wind without Integrated Battery Storage": "wind_mw",
        "Net Generation (MW) from Wind with Integrated Battery Storage": "wind_battery_mw",
        "Net Generation (MW) from Battery Storage": "battery_storage_mw",
        "Net Generation (MW) from Other Energy Storage": "other_storage_mw",
        "Net Generation (MW) from Unknown Energy Storage": "unknown_storage_mw",
        "Net Generation (MW) from Geothermal": "geothermal_mw",
        "Net Generation (MW) from Other Fuel Sources": "other_fuels_mw",
        "Net Generation (MW) from Unknown Fuel Sources": "unknown_fuels_mw",

        "Region": "region"
    })

    # 3) Zeitspalten in echte Datetime-Typen überführen.
    df = df.with_columns([
        pl.col("date")
        .str.strptime(pl.Datetime, format="%m/%d/%Y")
        .alias("date"),
        pl.col("utc_time_end")
        .str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p")
        .alias("utc_time_end"),
        pl.col("local_time_end")
        .str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p")
        .alias("local_time_end")
    ])

    # 4) Analysezeitraum eingrenzen.
    df = df.filter(
        (pl.col("utc_time_end") < pl.datetime(2026, 2, 18, 1, 0, 0)) &
        (pl.col("utc_time_end") > pl.datetime(2025, 7, 2, 1, 0, 0))
    )

    # 5) Fehlerfeatures für Prognosequalität ableiten.
    df = df.with_columns(
        (pl.col("demand_mw") - pl.col("demand_forecast_mw")).alias("forecast_error")
    ).with_columns(
        pl.when(pl.col("demand_mw") != 0)
        .then(pl.col("forecast_error") / pl.col("demand_mw"))
        .otherwise(0)
        .alias("percentage_forecast_error")
    )

    # 6) Brennstoffspalten für physikalische Konsistenzprüfung definieren.
    fuel_cols = [
        # Fossil
        "coal_mw",
        "natural_gas_mw",
        "petroleum_mw",

        # Erneuerbar
        "solar_mw",
        "solar_battery_mw",
        "wind_mw",
        "wind_battery_mw",
        "hydro_mw",

        # Speicher und unbekannte Erzeugung
        "nuclear_mw",
        "pumped_storage_mw",
        "battery_storage_mw",
        "other_storage_mw",
        "unknown_storage_mw",
        "geothermal_mw",
        "other_fuels_mw",
        "unknown_fuels_mw",
    ]

    # Alle Spalten vorher auf Float casten, falls Strings drin sind
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in fuel_cols + ["net_generation_mw"]])

    # 7) Gesamterzeugung aus Brennstoffkomponenten gegen gemeldete Erzeugung prüfen.
    df = df.with_columns([
        pl.sum_horizontal(fuel_cols).alias("fuel_sum"),
        (pl.sum_horizontal(fuel_cols) - pl.col("net_generation_mw")).alias("generation_diff")
    ])

    # Berechne für demand_mw den Modifizierten Z-Score
    df = df.with_columns(
        modified_z_score(pl.col("demand_mw")).alias("demand_mw_mzscore")
    )

    # 8) Datenintegrität prüfen und final sortieren.
    if len(check_duplicates(df)) > 0:
        raise ValueError("Daten enthalten Duplikate")
    
    df = df.sort("utc_time_end", descending=False)
    
    return df, fuel_cols

def get_interchange_data():
    """Lädt, normalisiert und validiert Interchange-Daten.

    Returns:
        Aufbereitetes Interchange-DataFrame, sortiert nach Datum und Stunde.

    Raises:
        ValueError: Falls doppelte Datensätze erkannt werden.
    """
    # Einheitliches Polars-Tabellenverhalten.
    pl.Config.set_tbl_rows(-1)
    
    # 1) Rohdaten laden und Felder harmonisieren.
    df = pl.read_parquet(INTERCHANGE_DATA_PATH)

    df = df.rename({
        "Balancing Authority": "balancing_authority",
        "Data Date": "date",
        "Hour Number": "hour",
        "Local Time at End of Hour": "local_time_end",
        "UTC Time at End of Hour": "utc_time_end",

        "Directly Interconnected Balancing Authority": "directly_interconnected_balancing_authority",
        "Interchange (MW)": "interchange_mw",

        "Region": "region",
        "DIBA_Region": "diba_region",
    })

    # 2) Zeitfelder normalisieren.
    df = df.with_columns([
        pl.col("date")
        .str.strptime(pl.Datetime, format="%m/%d/%Y")
        .alias("date"),
        pl.col("utc_time_end")
        .str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p")
        .alias("utc_time_end"),
        pl.col("local_time_end")
        .str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p")
        .alias("local_time_end")
    ])

    # 3) Analysefenster begrenzen.
    df = df.filter(
        (pl.col("utc_time_end") < pl.datetime(2026, 2, 18, 1, 0, 0)) &
        (pl.col("utc_time_end") > pl.datetime(2026, 2, 18, 1, 0, 0))
    )

    # 4) Integritätsprüfung und Sortierung.
    if len(check_duplicates(df)) > 0:
        raise ValueError("Daten enthalten Duplikate")
    
    df = df.sort(["date", "hour"], descending=False)
    
    return df
    
def get_subregion_data():
    """Lädt, normalisiert und validiert Subregion-Nachfragedaten.

    Returns:
        Aufbereitetes Subregion-DataFrame, sortiert nach Datum und Stunde.

    Raises:
        ValueError: Falls doppelte Datensätze erkannt werden.
    """
    # Einheitliches Polars-Tabellenverhalten.
    pl.Config.set_tbl_rows(-1)
    
    # 1) Rohdaten laden und Felder harmonisieren.
    df = pl.read_parquet(SUBREGION_DATA_PATH)

    df = df.rename({
        "Balancing Authority": "balancing_authority",
        "Data Date": "date",
        "Hour Number": "hour",
        "Local Time at End of Hour": "local_time_end",
        "UTC Time at End of Hour": "utc_time_end",

        "Demand (MW)": "demand_mw",

        "Sub-Region": "sub_region",
    })

    # 2) Zeitfelder normalisieren.
    df = df.with_columns([
        pl.col("date")
        .str.strptime(pl.Datetime, format="%m/%d/%Y")
        .alias("date"),
        pl.col("utc_time_end")
        .str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p")
        .alias("utc_time_end"),
        pl.col("local_time_end")
        .str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p")
        .alias("local_time_end")
    ])

    # 3) Analysefenster begrenzen.
    df = df.filter(
        (pl.col("utc_time_end") < pl.datetime(2026, 2, 18, 1, 0, 0)) &
        (pl.col("utc_time_end") > pl.datetime(2026, 2, 18, 1, 0, 0))
    )
    
    # 4) Integritätsprüfung und Sortierung.
    if len(check_duplicates(df)) > 0:
        raise ValueError("Daten enthalten Duplikate")
    
    df = df.sort(["date", "hour"], descending=False)
    
    return df

# Hilfsfunktionen
def check_duplicates(df: pl.DataFrame):
    """Findet doppelte Zeilen je Balancing Authority und Zeitstempel.

    Args:
        df: DataFrame, das die Spalten ``balancing_authority`` und
            ``utc_time_end`` enthalten sollte.

    Returns:
        DataFrame mit doppelten Schlüsselkombinationen und deren Häufigkeit.
    """
    # Doppelte Schlüssel werden gezählt und nur bei Mehrfachvorkommen ausgegeben.
    return (
        df.group_by(["balancing_authority", "utc_time_end"])
        .len()
        .filter(pl.col("len") > 1)
    )

def modified_z_score(col: pl.Expr):
    """Erzeugt eine Polars-Expression zur Berechnung des Modified Z-Scores.

    Args:
        col: Polars-Spaltenexpression.

    Returns:
        Expression, die den Modified Z-Score je Zeile liefert.
    """
    # Robuste Lage- und Streuungsmaße für ausreißerresistente Skalierung.
    median = col.median()
    mad = (col - median).abs().median()
    return 0.6745 * (col - median) / mad

def csv_to_parquet(csv_path: str, parquet_path: str, infer_schema_length: int = 1000):
    """Konvertiert eine CSV-Datei ins Parquet-Format.

    Args:
        csv_path: Pfad zur Eingabe-CSV-Datei.
        parquet_path: Pfad, unter dem die Parquet-Datei gespeichert wird.
        infer_schema_length: Anzahl Zeilen für die Schema-Inferenz.

    Returns:
        ``None``. Schreibt die Parquet-Datei auf die Festplatte.
    """
    # CSV einlesen und typische fehlende Werte explizit behandeln.
    df = pl.read_csv(
        csv_path,
        infer_schema_length=infer_schema_length,  # robustes Schema
        null_values=["", "NA", "NaN"]  # häufige fehlende Werte
    )

    # Parquet speichern
    df.write_parquet(parquet_path)
    print(f"Parquet-Datei gespeichert unter: {parquet_path}")
