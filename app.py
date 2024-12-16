from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('salary_model.pkl')

@app.route('/')
def home():
    return "Salary Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    
    if 'YearsExperience' not in data:
        return jsonify({"error": "Missing 'YearsExperience' in request"}), 400

    try:
        
        years_experience = float(data['YearsExperience'])
        prediction = model.predict(np.array([[years_experience]])) 

        
        return jsonify({"YearsExperience": years_experience, "PredictedSalary": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
