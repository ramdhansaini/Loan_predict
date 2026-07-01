from flask import Flask, render_template, request
import pickle
import os
import numpy as np
app = Flask(__name__, template_folder="Templates", static_folder="Static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "loan.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

with open(model_path,"rb") as f:
    model = pickle.load(f)
with open(scaler_path,"rb") as f:
    scaler = pickle.load(f)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    income = int(request.form["income"])
    loan_amount = int(request.form["loan_amount"])
    # Add more form fields as needed
    # Example:
    # Person_emp_length = request.form["Person_emp_length"]
    # Person_home_ownership = request.form["Person_home_ownership"]
    # Credit_score = int(request.form["Credit_score"])

    # Create a numpy array with the input values
    input_data = np.array([[age, income, loan_amount]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        loan_status = "Approved"
    else:
        loan_status = "Not Approved"

    return render_template("result.html", loan_status=loan_status)

if __name__ =="__main__":
    app.run(debug=True)
    
