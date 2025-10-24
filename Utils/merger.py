import os
import pandas as pd
import glob

def merge_csv_files():
    # Set fixed directory path
    csv_dir = r"\leakmap\Csv"
    
    # Get all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return
    
    print(f"Found {len(csv_files)} CSV file(s): {[os.path.basename(f) for f in csv_files]}")
    
    # List to store all dataframes
    dataframes = []
    
    # Read each CSV file
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"Loaded {file} with {len(df)} rows")
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        print("No valid CSV files to merge.")
        return
    
    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save to merged.csv in the same directory
    output_path = os.path.join(csv_dir, "merged.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Successfully merged {len(dataframes)} files into {output_path} with {len(merged_df)} total rows")

if __name__ == "__main__":
    merge_csv_files()
