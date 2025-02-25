from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\91636\OneDrive\Desktop\battery_python\model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Define API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.json  
        features = pd.DataFrame([data['features']], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

        # Make prediction
        prediction = model.predict(features)

        # Return result
        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)  # Run API locally
