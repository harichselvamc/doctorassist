from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model_file_path = 'random_forest_model.pkl'
scaler_file_path = 'scaler.pkl'
with open(model_file_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the recommendation function
def recommend(source):
    if source == 1:
        return "Maintain a balanced diet and consult a doctor for regular checkups."
    elif source == 0:
        return "Your health parameters are stable. Continue with your current lifestyle."
    else:
        return "Consider a detailed medical examination for potential issues."

# Define the routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input data from the form
            input_data = [
                float(request.form['HAEMATOCRIT']),
                float(request.form['HAEMOGLOBINS']),
                float(request.form['ERYTHROCYTE']),
                float(request.form['LEUCOCYTE']),
                float(request.form['THROMBOCYTE']),
                float(request.form['MCH']),
                float(request.form['MCHC']),
                float(request.form['MCV']),
                float(request.form['AGE'])
            ]

            # Transform the input data
            input_data = np.array([input_data])
            input_scaled = scaler.transform(input_data)

            # Make predictions
            prediction = loaded_model.predict(input_scaled)[0]
            recommendation = recommend(prediction)

            # Display results on the same page
            return render_template('index.html', 
                                   prediction='At Risk' if prediction == 1 else 'Healthy', 
                                   recommendation=recommendation)

        except Exception as e:
            return render_template('index.html', prediction="Error", recommendation=str(e))

    # Initial GET request
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
