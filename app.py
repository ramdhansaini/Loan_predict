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
    person_ownership = request.form["person_home_ownership"]
    person_emp_length = int(request.form["person_emp_length"])
    loan_amount = int(request.form["loan_amount"])
    loan_int_rate = float(request.form["loan_int_rate"])
    loan_intent = request.form["loan_intent"]
    loan_grade = request.form["loan_grade"]
    loan_percent_income = float(request.form["loan_percent_income"])
    cb_person_default_on_file = request.form["cb_person_default_on_file"]
    cb_person_cred_hist_length = int(request.form["cb_person_cred_hist_length"])
    
    # Add more form fields as needed
    # Example:
    #  could not convert string to float: np.str_('')

    # Create a numpy array with the input values
    input_data = np.array([[age, income, loan_amount, loan_int_rate, loan_intent, loan_grade, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length  ]])

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
    
