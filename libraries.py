# Import necessary libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --------------------------------------
# Step 1: Load CSV Files from Folder
# --------------------------------------

folder_path = r"C:\Users\91636\Downloads\battery datasets\cleaned_dataset\data"

if os.path.exists(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    if not files:
        print("Error: No CSV files found in the folder.")
        exit()  # Exit if no files are found

    all_data = []  # List to store DataFrames
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        all_data.append(df)
        print(f"Loaded {file} with shape {df.shape}")

    # Merge all CSV files into one DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    print("Final dataset shape:", final_df.shape)

    # Save merged dataset
    final_df.to_csv("merged_dataset.csv", index=False)
else:
    print("Error: The specified folder path does not exist.")
    exit()  # Exit if folder doesn't exist

# --------------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# --------------------------------------

# Load the merged dataset
df = pd.read_csv("merged_dataset.csv")

# Display basic information
print(df.info())  # Check column names and types
print(df.head())  # Display first 5 rows
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Handle missing values (Drop rows with missing values)
df.dropna(inplace=True)
print("After removing missing values, dataset shape:", df.shape)

# Summary statistics
print(df.describe())

# --------------------------------------
# Step 3: Data Visualization
# --------------------------------------

# 1️⃣ Histogram of all numeric features
df.hist(figsize=(10, 8))
plt.show()

# 2️⃣ Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 3️⃣ Boxplot to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot for Outlier Detection")
plt.show()

# --------------------------------------
# Step 4: Feature Engineering
# --------------------------------------

# Identify categorical columns dynamically
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

if categorical_columns:
    print("Categorical columns detected:", categorical_columns)
    df = pd.get_dummies(df, columns=categorical_columns)
else:
    print("No categorical columns found.")

# --------------------------------------
# Step 5: Splitting the Data
# --------------------------------------

# Automatically detect the target column (assumed to be the last column)
target_column = df.columns[-1]  # Assuming the last column is the target

if target_column in df.columns:
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print(f"Target column '{target_column}' selected.")
else:
    print(f"Error: Target column '{target_column}' not found in dataset.")
    exit()

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------
# Step 6: Model Training & Evaluation
# --------------------------------------

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# --------------------------------------
# Step 7: Save the Trained Model
# --------------------------------------

joblib.dump(model, "model.pkl")
print("Model saved as 'model.pkl'")

# Save test results
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv("test_results.csv", index=False)
print("Test results saved as 'test_results.csv'")

# Load and Test Saved Model (To Validate)
loaded_model = joblib.load("model.pkl")
test_pred = loaded_model.predict(X_test)
print(f"Loaded Model Test Accuracy: {accuracy_score(y_test, test_pred):.2f}")
