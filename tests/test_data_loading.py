"""Test loading actual data to verify compatibility."""
import polars as pl
from pathlib import Path

def test_data_loading():
    """Test loading actual data files."""
    base_dir = Path(".")
    
    datasets = {
        "eicu": base_dir / "data" / "eicu",
        "mimic": base_dir / "data" / "mimic",
    }
    
    print("="*80)
    print("DATA LOADING TEST")
    print("="*80)
    
    for dataset_name, data_dir in datasets.items():
        print(f"\n{'-'*80}")
        print(f"{dataset_name.upper()}")
        print(f"{'-'*80}")
        
        # Test loading each file
        files_to_test = {
            "dynamic": "dyn.parquet",
            "static": "sta.parquet",
            "outcome": "outc.parquet",
        }
        
        # Handle MIMIC static file naming
        if dataset_name == "mimic":
            static_alt = data_dir / "sta (1).parquet"
            if static_alt.exists():
                files_to_test["static"] = "sta (1).parquet"
        
        for file_type, filename in files_to_test.items():
            filepath = data_dir / filename
            if not filepath.exists():
                print(f"  {file_type}: File not found: {filename}")
                continue
            
            try:
                df = pl.read_parquet(filepath)
                print(f"  {file_type}: [OK] Loaded successfully")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {len(df.columns)}")
                
                # Check for required columns
                if "stay_id" in df.columns:
                    unique_stays = df["stay_id"].n_unique()
                    print(f"    Unique stay_ids: {unique_stays}")
                
                # Test time column if present
                if "time" in df.columns:
                    print(f"    Time column dtype: {df['time'].dtype}")
                    # Try to parse time if it's a string
                    if df["time"].dtype == pl.String:
                        try:
                            # Try to extract hours from "0 days HH:MM:SS" format
                            sample = df["time"].head(1).item()
                            print(f"    Time sample: '{sample}'")
                            # Check if we can extract numeric value
                            if "days" in sample and ":" in sample:
                                # Extract hours part
                                parts = sample.split()
                                if len(parts) >= 2:
                                    time_part = parts[2]  # "HH:MM:SS"
                                    hours = int(time_part.split(":")[0])
                                    print(f"    Parsed hours: {hours}")
                        except Exception as e:
                            print(f"    Time parsing test failed: {e}")
                
                # Test label column if present
                if "label" in df.columns:
                    print(f"    Label column dtype: {df['label'].dtype}")
                    label_counts = df["label"].value_counts().sort("label")
                    print(f"    Label distribution:")
                    for row in label_counts.iter_rows(named=True):
                        print(f"      Label {row['label']}: {row['count']} ({row['count']/len(df)*100:.2f}%)")
                
            except Exception as e:
                print(f"  {file_type}: [ERROR] Error loading: {e}")
                import traceback
                traceback.print_exc()
        
        # Test joining dynamic and static data (common operation)
        print(f"\n  Testing data joins:")
        try:
            dyn_path = data_dir / "dyn.parquet"
            sta_path = data_dir / files_to_test["static"]
            
            if dyn_path.exists() and sta_path.exists():
                dyn_df = pl.read_parquet(dyn_path)
                sta_df = pl.read_parquet(sta_path)
                
                if "stay_id" in dyn_df.columns and "stay_id" in sta_df.columns:
                    # Try a simple join
                    joined = dyn_df.join(sta_df, on="stay_id", how="left")
                    print(f"    Dynamic + Static join: [OK] Success")
                    print(f"      Joined shape: {joined.shape}")
                    print(f"      Original dynamic shape: {dyn_df.shape}")
                else:
                    print(f"    Join test skipped: missing stay_id column")
        except Exception as e:
            print(f"    Join test: [ERROR] Error: {e}")

if __name__ == "__main__":
    test_data_loading()

