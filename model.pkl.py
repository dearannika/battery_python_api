import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a simple model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 4: Save the trained model
model_path = r"C:\Users\91636\OneDrive\Desktop\battery_python\model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved successfully as {model_path}")

# Step 5: Load the saved model
with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)

# Step 6: Make predictions
predictions = loaded_model.predict(X_test)

# Step 7: Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 8: Display sample predictions
print("Sample Predictions:", predictions[:5])
import pickle
import pandas as pd

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Example input for prediction
sample_input = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your actual input
prediction = model.predict(sample_input)

print(f"Prediction: {prediction[0]}")
sample_input = [[6.2, 3.4, 5.4, 2.3]]  # Example new input
prediction = model.predict(sample_input)
print(f"New Prediction: {prediction[0]}")
