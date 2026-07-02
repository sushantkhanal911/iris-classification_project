from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
# CORS allows your frontend (running in a browser) to communicate securely with your backend
CORS(app) 

# Load the saved decision tree model
model = joblib.load('decisiontree_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the JSON data sent from the frontend
        data = request.get_json()
        
        # 2. Extract the 4 standard features
        sl = float(data['sepal_length'])
        sw = float(data['sepal_width'])
        pl = float(data['petal_length'])
        pw = float(data['petal_width'])
        
        # 3. Calculate Petal Area
        petal_area = pl * pw
        
        # 4. Create the array for the model
        features = np.array([[sl, sw, pl, pw, petal_area]])
        
        # 5. Make the prediction (returns a numpy.int64 array, e.g., [0])
        prediction_encoded = model.predict(features)
        
        # FIX: Convert numpy.int64 to a standard Python int
        predicted_class = int(prediction_encoded[0])
        
        # 6. Map the number back to the original flower name
        species_mapping = {
            0: "Iris-setosa",
            1: "Iris-versicolor",
            2: "Iris-virginica"
        }
        prediction_label = species_mapping.get(predicted_class, "Unknown Species")
        
        # 7. Send the string back to the frontend safely
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)