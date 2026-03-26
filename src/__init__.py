

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split        
from sklearn.linear_model import LinearRegression



def load_data(location):
    """Load the dataset for the specified location.
    Takes inputs like "1", "2", "3", "4" or "Location1",
      "Location2", etc. and returns the corresponding DataFrame.
    Also, changes the "Time" column to datetime if it exists.
    """
    normalized = str(location).strip().lower().replace("location", "")
    if normalized not in {"1", "2", "3", "4"}:
        raise ValueError("location must be one of: 1, 2, 3, 4 or Location1-Location4")

    base_dir = Path(__file__).resolve().parent.parent / "inputs"
    filepath = base_dir / f"Location{normalized}.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"Could not find {filepath}")

    df = pd.read_csv(filepath)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    return df


def power_plot(location, year):
    """Plot and save Power time series for a selected location and 
    starting time (year).

    The PNG is saved to outputs/ as power_locationX_YYYY.png.
    """
    df = load_data(location)

    if "Time" not in df.columns:
        raise KeyError("Expected a 'Time' column in the dataset")
    if "Power" not in df.columns:
        raise KeyError("Expected a 'Power' column in the dataset")

    year_int = int(year)
    start = pd.Timestamp(year=year_int, month=1, day=1)
    end = pd.Timestamp(year=year_int, month=12, day=31, hour=23, minute=59, second=59)

    mask = (df["Time"] >= start) & (df["Time"] <= end)
    df_year = df.loc[mask, ["Time", "Power"]].dropna()

    if df_year.empty:
        raise ValueError(f"No Power data found for location {location} in year {year_int}")

    normalized_location = str(location).strip().lower().replace("location", "")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_year["Time"], df_year["Power"], color="purple", linewidth=1.8, alpha=0.95)
    ax.set_title(f"Power Time Series - Location {normalized_location}, {year_int}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"power_location{normalized_location}_{year_int}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()

    return fig, ax, output_path


def data_split(location, test_size=0.2, random_state=42):
    """Split the dataset for the specified location into training and testing sets.
    Returns X_train, X_test, y_train, y_test.

    The function uses an 80/20 train-test split by default 
    with random_state=42. You can override these defaults by 
    passing different values for test_size and random_state.
    """
    df = load_data(location)

    if "Power" not in df.columns:
        raise KeyError("Expected a 'Power' column in the dataset")

    X = df.drop(columns=["Power"], errors="ignore")
    y = df["Power"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test





