import unittest
import joblib
import numpy as np
from app import app

class TestSalaryPredictionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
    
        cls.model = joblib.load('salary_model.pkl')

    def test_model_prediction(self):
        
        input_data = np.array([[5]]) 
        prediction = self.model.predict(input_data)
        self.assertTrue(prediction[0] > 0, "Prediction should be greater than 0 for valid input.")

    def test_model_input_shape(self):
    
        invalid_data = np.array([5])  
        with self.assertRaises(ValueError):
            self.model.predict(invalid_data)

class TestFlaskAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
    
        cls.client = app.test_client()

    def test_home_endpoint(self):
        
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Salary Prediction API", response.data)

    def test_predict_endpoint_valid(self):
        
        response = self.client.post('/predict', json={"YearsExperience": 5})
        self.assertEqual(response.status_code, 200)
        self.assertIn("PredictedSalary", response.json)

    def test_predict_endpoint_invalid(self):
    
        response = self.client.post('/predict', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

if __name__ == '__main__':
    unittest.main()
