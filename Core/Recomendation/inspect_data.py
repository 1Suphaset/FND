import pandas as pd
import joblib
import os

def print_separator():
    print("\n" + "="*50 + "\n")

# Check CSV data
print("Checking CSV data:")
csv_path = os.path.join('..', 'RawData', 'NewsData10200records.csv')
try:
    csv_data = pd.read_csv(csv_path)
    print("CSV Columns:", csv_data.columns.tolist())
    print("\nFirst row of CSV:")
    print(csv_data.iloc[0])
except Exception as e:
    print(f"Error reading CSV: {e}")

print_separator()

# Check pickle data
print("Checking pickle data:")
pkl_path = os.path.join('..', 'InitData', 'result', 'X_under.pkl')
try:
    pkl_data = joblib.load(pkl_path)
    if isinstance(pkl_data, pd.DataFrame):
        print("Pickle Columns:", pkl_data.columns.tolist())
        print("\nFirst row of pickle data:")
        print(pkl_data.iloc[0])
    else:
        print("Pickle data type:", type(pkl_data))
        print("Data structure:", pkl_data)
except Exception as e:
    print(f"Error reading pickle: {e}")