import pandas as pd

class DataProcessor:
    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    @staticmethod
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and format."""
        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]

        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
        elif "date" in df.columns:
            df.rename(columns={"date": "timestamp"}, inplace=True)

        for col in DataProcessor.REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        df = df[DataProcessor.REQUIRED_COLUMNS]
        
        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def save_to_csv(df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)