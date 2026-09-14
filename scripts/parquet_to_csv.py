import sys
import pandas as pd
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python parquet_to_csv.py <path_to_parquet_file>")
        sys.exit(1)
        
    parquet_path = Path(sys.argv[1])
    if not parquet_path.exists():
        print(f"Error: File '{parquet_path}' does not exist.")
        sys.exit(1)
        
    csv_path = parquet_path.with_suffix('.csv')
    
    try:
        print(f"Reading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        print(f"Saving to {csv_path}...")
        df.to_csv(csv_path, index=False)
        
        print("Done!")
        print(f"Rows: {len(df)}")
        print("\nPreview:")
        print(df.head())
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    main()
