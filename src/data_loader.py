from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Optional

import pandas as pd


REQUIRED_COLUMNS = {"date", "time", "symbol", "open", "high", "low", "close", "volume"}

# Column name mapping for different data sources
COLUMN_MAPPINGS = {
    # Common Excel/CSV variations
    'Date': 'date',
    'DATE': 'date',
    'Datetime': 'datetime',
    'DateTime': 'datetime',
    'DATETIME': 'datetime',
    'Timestamp': 'datetime',
    'timestamp': 'datetime',
    'TIMESTAMP': 'datetime',
    'Time': 'time',
    'TIME': 'time',
    'Open': 'open',
    'OPEN': 'open',
    'High': 'high',
    'HIGH': 'high',
    'Low': 'low',
    'LOW': 'low',
    'Close': 'close',
    'CLOSE': 'close',
    'Volume': 'volume',
    'VOLUME': 'volume',
    'Symbol': 'symbol',
    'SYMBOL': 'symbol',
    'Instrument': 'symbol',
    'INSTRUMENT': 'symbol',
}


def _clean_date(value: str) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip()
    cleaned = cleaned.replace('="', "").replace('"', "").replace("=", "")
    return cleaned


def _standardize_columns(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    """
    Standardize column names using the mapping dictionary.
    
    Args:
        df: DataFrame with potentially non-standard column names
        source_file: Name of source file for error messages
    
    Returns:
        DataFrame with standardized column names
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Apply column mapping
    df.columns = [COLUMN_MAPPINGS.get(col, col.lower()) for col in df.columns]
    
    # Check for required columns
    required_base = {"open", "high", "low", "close"}
    available = set(df.columns)
    
    # Check if we have datetime or date+time
    has_datetime = "datetime" in available
    has_date_time = ("date" in available and "time" in available)
    
    if not has_datetime and not has_date_time:
        print(f"Warning: No datetime column found in {source_file}")
        print(f"Available columns: {list(df.columns)}")
        print("Attempting to use index as datetime...")
    
    # Check OHLC columns
    missing_ohlc = required_base - available
    if missing_ohlc:
        raise ValueError(
            f"Missing required OHLC columns in {source_file}: {missing_ohlc}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please ensure file has: Open, High, Low, Close (case-insensitive)"
        )
    
    # Add volume and symbol if missing
    if "volume" not in df.columns:
        print(f"Warning: 'volume' column not found in {source_file}, using default value 0")
        df["volume"] = 0
    
    if "symbol" not in df.columns:
        print(f"Warning: 'symbol' column not found in {source_file}, using default 'UNKNOWN'")
        df["symbol"] = "UNKNOWN"
    
    return df


def _load_single_excel(path: Path) -> pd.DataFrame:
    """
    Load a single Excel file with validation and column mapping.
    
    Args:
        path: Path to Excel file
    
    Returns:
        DataFrame with standardized columns and datetime index
    """
    print(f"Loading Excel file: {path.name}")
    
    # Read Excel file
    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {path.name}: {e}")
    
    if df.empty:
        raise ValueError(f"Excel file {path.name} is empty")
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Original columns: {list(df.columns)}")
    
    # Standardize column names
    df = _standardize_columns(df, path.name)
    print(f"  Standardized columns: {list(df.columns)}")
    
    # Handle datetime index
    if "datetime" in df.columns:
        # Already have datetime column
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns and "time" in df.columns:
        # Combine date and time
        print(f"  Combining date and time columns...")
        # Handle date - try multiple approaches
        try:
            df["date_str"] = df["date"].astype(str)
            df["time_str"] = df["time"].astype(str)
            # Try with space separator first
            df["datetime"] = pd.to_datetime(
                df["date_str"] + " " + df["time_str"],
                errors="coerce",
            )
            # If that didn't work, try cleaning the date first
            if df["datetime"].isna().all():
                df["date_clean"] = df["date"].map(_clean_date)
                df["datetime"] = pd.to_datetime(
                    df["date_clean"] + " " + df["time_str"],
                    errors="coerce",
                )
        except Exception as e:
            print(f"  Error combining date and time: {e}")
            # Fallback: try date only
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    elif "date" in df.columns:
        # Only date column
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # Try to use index if it's datetime-like
        if isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = df.index
        else:
            raise ValueError(
                f"Cannot determine datetime column in {path.name}. "
                f"Expected 'Date', 'Datetime', or 'Timestamp' column."
            )
    
    # Clean and set index
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime")
    
    # Remove duplicates based on datetime - keep last
    df_len_before = len(df)
    df = df[~df["datetime"].duplicated(keep="last")].copy()
    if len(df) < df_len_before:
        print(f"  Removed {df_len_before - len(df):,} duplicate timestamps")
    
    df = df.set_index("datetime")
    
    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    
    print(f"  ✓ Processed {len(df):,} valid rows with datetime index")
    
    return df
def _load_single_csv(path: Path) -> pd.DataFrame:
    """Load a single CSV file with column mapping support."""
    df = pd.read_csv(path)
    
    # Standardize column names
    df = _standardize_columns(df, path.name)
    
    # Handle datetime
    if "datetime" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["date"] = df["date"].map(_clean_date)
            df["datetime"] = pd.to_datetime(
                df["date"] + " " + df["time"].astype(str),
                format="%d-%m-%y %H:%M:%S",
                errors="coerce",
            )
    
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    return df


def load_csv_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frames.append(_load_single_csv(path))
    if not frames:
        raise ValueError("No CSV files found for loading.")
    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def load_data(source_path: str) -> pd.DataFrame:
    """
    Load financial data from CSV, Excel, or directory of files.
    
    Supports:
    - Single CSV file
    - Single Excel file (.xlsx, .xls)
    - Directory containing multiple CSV files
    
    Args:
        source_path: Path to file or directory
    
    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    path = Path(source_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {source_path}")
    
    if path.is_dir():
        # Load all CSV files from directory
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {source_path}")
        print(f"Found {len(csv_files)} CSV files in directory")
        return load_csv_files(csv_files)
    
    if path.is_file():
        # Check file extension
        if path.suffix.lower() in ['.xlsx', '.xls']:
            return _load_single_excel(path)
        elif path.suffix.lower() == '.csv':
            return _load_single_csv(path)
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}\n"
                f"Supported formats: .csv, .xlsx, .xls"
            )
    
    raise FileNotFoundError(f"Invalid path: {source_path}")


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate that loaded data meets requirements for backtesting.
    
    Args:
        df: DataFrame to validate
    
    Raises:
        ValueError: If data is invalid
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for NaN values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        print("Warning: Found NaN values in data:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} NaN values")
    
    # Check data ranges
    if (df["high"] < df["low"]).any():
        print("Warning: Found rows where High < Low (data quality issue)")
    
    if (df["close"] > df["high"]).any() or (df["close"] < df["low"]).any():
        print("Warning: Found rows where Close is outside High-Low range")
    
    print(f"\n✓ Data validation passed: {len(df):,} rows, {df.index[0]} to {df.index[-1]}")
