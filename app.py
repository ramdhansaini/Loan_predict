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

HOME_OWNERSHIP_MAP = {
    "MORTGAGE": 0,
    "OTHER": 1,
    "OWN": 2,
    "RENT": 3,
}
LOAN_INTENT_MAP = {
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5,
}
LOAN_GRADE_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
}
DEFAULT_ON_FILE_MAP = {"N": 0, "Y": 1}

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    income = int(request.form["income"])
    person_ownership = HOME_OWNERSHIP_MAP.get(request.form.get("person_home_ownership", ""))
    person_emp_length = int(request.form["person_emp_length"])
    loan_amount = float(request.form["loan_amount"])
    loan_int_rate = float(request.form["loan_int_rate"])
    loan_intent = LOAN_INTENT_MAP.get(request.form.get("loan_intent", ""))
    loan_grade = LOAN_GRADE_MAP.get(request.form.get("loan_grade", ""))
    loan_percent_income = float(request.form["loan_percent_income"])
    cb_person_default_on_file = DEFAULT_ON_FILE_MAP.get(request.form.get("cb_person_default_on_file", ""))
    cb_person_cred_hist_length = int(request.form["cb_person_cred_hist_length"])

    if None in (person_ownership, loan_intent, loan_grade, cb_person_default_on_file):
        return render_template("index.html", error="Please complete all dropdown selections.")

    input_data = np.array([[
        age,
        income,
        person_ownership,
        person_emp_length,
        loan_intent,
        loan_grade,
        loan_amount,
        loan_int_rate,
        loan_percent_income,
        cb_person_default_on_file,
        cb_person_cred_hist_length,
    ]], dtype=float)

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
    
