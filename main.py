from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# Load data and model
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel_v142.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    # Get unique sorted locations for dropdown
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('sqft'))

    print(location, bhk, bath, sqft)

    # Prepare input for model
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Make prediction
    prediction = pipe.predict(input_df)[0]

    # Return formatted prediction
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
