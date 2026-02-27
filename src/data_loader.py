import polars as pl
from .config import BALANCE_DATA_PATH, INTERCHANGE_DATA_PATH, SUBREGION_DATA_PATH

def get_balance_data():
    pl.Config.set_tbl_rows(-1)
    
    df = pl.read_parquet(BALANCE_DATA_PATH)

    # Alle Spalten rausfiltern, die _imputed oder _adjusted enthalten
    cols_to_keep = [c for c in df.columns if not (c.endswith("(Imputed)") or c.endswith("(Adjusted)"))]
    df = df.select(cols_to_keep)

    # Spaltennamen in einheitliches Format bringen
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

    # Daten in DateTime konvertieren
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

    # Alle Daten vor 02.07.2025 und nach 17.02.2026 abschneiden (Es gibt da nur noch Forecast-Daten)
    df = df.filter(
        (pl.col("utc_time_end") < pl.datetime(2026, 2, 18, 1, 0, 0)) &
        (pl.col("utc_time_end") > pl.datetime(2025, 7, 2, 1, 0, 0))
    )

    df = df.with_columns(
        (pl.col("demand_mw") - pl.col("demand_forecast_mw")).alias("forecast_error")
    ).with_columns(
        pl.when(pl.col("demand_mw") != 0)
        .then(pl.col("forecast_error") / pl.col("demand_mw"))
        .otherwise(0)
        .alias("percentage_forecast_error")
    )

    # Physikalische Konsistenz prüfen
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

    # Summieren und Differenz in einem Schritt
    df = df.with_columns([
        pl.sum_horizontal(fuel_cols).alias("fuel_sum"),
        (pl.sum_horizontal(fuel_cols) - pl.col("net_generation_mw")).alias("generation_diff")
    ])

    # Berechne für demand_mw den Modifizierten Z-Score
    df = df.with_columns(
        modified_z_score(pl.col("demand_mw")).alias("demand_mw_mzscore")
    )

    # Kontolliere auf Duplikate
    if len(check_duplicates(df)) > 0:
        raise ValueError("Daten enthalten Duplikate")
    
    df = df.sort("utc_time_end", descending=False)
    
    return df, fuel_cols

def get_interchange_data():
    pl.Config.set_tbl_rows(-1)
    
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

    # Daten in DateTime konvertieren
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

    # Alle Daten nach 17.02.2026 abschneiden (Es gibt da nur noch Forecast-Daten)
    df = df.filter(
        (pl.col("utc_time_end") < pl.datetime(2026, 2, 18, 1, 0, 0)) &
        (pl.col("utc_time_end") > pl.datetime(2026, 2, 18, 1, 0, 0))
    )

    # Kontolliere auf Duplikate
    if len(check_duplicates(df)) > 0:
        raise ValueError("Daten enthalten Duplikate")
    
    df = df.sort(["date", "hour"], descending=False)
    
    return df
    
def get_subregion_data():
    pl.Config.set_tbl_rows(-1)
    
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

    # Daten in DateTime konvertieren
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

    # Alle Daten nach 17.02.2026 abschneiden (Es gibt da nur noch Forecast-Daten)
    df = df.filter(
        (pl.col("utc_time_end") < pl.datetime(2026, 2, 18, 1, 0, 0)) &
        (pl.col("utc_time_end") > pl.datetime(2026, 2, 18, 1, 0, 0))
    )
    
    # Kontolliere auf Duplikate
    if len(check_duplicates(df)) > 0:
        raise ValueError("Daten enthalten Duplikate")
    
    df = df.sort(["date", "hour"], descending=False)
    
    return df

# Hilfsfunktionen
def check_duplicates(df: pl.DataFrame):
    return (
        df.group_by(["balancing_authority", "utc_time_end"])
        .len()
        .filter(pl.col("len") > 1)
    )

def modified_z_score(col: pl.Expr):
    median = col.median()
    mad = (col - median).abs().median()
    return 0.6745 * (col - median) / mad

def csv_to_parquet(csv_path: str, parquet_path: str, infer_schema_length: int = 1000):
    """
    Liest eine CSV-Datei ein und speichert sie als Parquet.

    Args:
        csv_path (str): Pfad zur Eingabe-CSV.
        parquet_path (str): Pfad zur Ausgabe-Parquet-Datei.
        infer_schema_length (int): Anzahl Zeilen für Schema-Erkennung. Standard 1000.
    """
    # CSV einlesen
    df = pl.read_csv(
        csv_path,
        infer_schema_length=infer_schema_length,  # robustes Schema
        null_values=["", "NA", "NaN"]  # häufige fehlende Werte
    )

    # Parquet speichern
    df.write_parquet(parquet_path)
    print(f"Parquet-Datei gespeichert unter: {parquet_path}")