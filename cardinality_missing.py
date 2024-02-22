import pandas as pd

# Load the dataset
data = pd.read_csv("data/properties.csv")

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Calculate missing value percentage for each column
missing_percentages = data.isnull().mean()

# Calculate cardinality for each categorical column
cardinality = {}
for col in categorical_columns:
    unique_values = data[col].nunique()
    cardinality[col] = unique_values / len(data)

# Set thresholds for high cardinality and high missing values
cardinality_threshold = 0.5  # Adjust as per your dataset and requirements
missing_threshold = 0.2  # Adjust as per your dataset and requirements

# Identify high cardinality columns
high_cardinality_columns = [col for col, card in cardinality.items() if card > cardinality_threshold]

# Identify columns with high missing values
high_missing_columns = [col for col, missing_percentage in missing_percentages.items() if missing_percentage > missing_threshold]

# Combine high cardinality and high missing value columns
high_cardinality_and_missing_columns = list(set(high_cardinality_columns) | set(high_missing_columns))

print("Columns with High Cardinality and/or High Missing Values:")
for col in high_cardinality_and_missing_columns:
    print(f"{col}: Cardinality = {cardinality.get(col, 0)}, Missing Percentage = {missing_percentages.get(col, 0)}")
