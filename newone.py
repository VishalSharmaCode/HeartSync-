from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("heart_model.sav", "rb"))

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get the input data from the request
    input_data = [data[col] for col in ['male', 'age', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
    input_data = np.asarray(input_data).reshape(1, -1)  # Reshape the data
    prediction = model.predict(input_data)[0]  # Make prediction
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
