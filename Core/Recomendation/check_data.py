import joblib
import os

# Load the data file
data_path = os.path.join('..', 'InitData', 'result', 'X_under.pkl')
data = joblib.load(data_path)

# Print column names
print("Available columns:", data.columns.tolist())
print("\nFirst row sample:")
print(data.iloc[0])