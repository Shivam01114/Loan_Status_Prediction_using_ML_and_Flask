from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ===============================
# Load trained model (Pipeline)
# ===============================
MODEL_PATH = os.path.join("model", "loan_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/result", methods=["POST"])
def result():
    # ===============================
    # Collect form data (SAFE CASTING)
    # ===============================
    try:
        data = [
            int(request.form.get('Gender', 0)),
            int(request.form.get('Married', 0)),
            int(request.form.get('Dependents', 0)),
            int(request.form.get('Education', 0)),
            int(request.form.get('Self_Employed', 0)),
            float(request.form.get('ApplicantIncome', 0)),
            float(request.form.get('CoapplicantIncome', 0)),
            float(request.form.get('LoanAmount', 0)),
            float(request.form.get('Loan_Amount_Term', 0)),
            float(request.form.get('Credit_History', 0)),
            int(request.form.get('Property_Area', 0))
        ]
    except Exception as e:
        print("❌ FORM DATA ERROR:", e)
        return "Invalid input data"

    # Convert to numpy array
    input_array = np.array(data).reshape(1, -1)

    # ===============================
    # MODEL PREDICTION
    # ===============================
    prediction = model.predict(input_array)

    # 🔴 DEBUG (VERY IMPORTANT)
    print("===================================")
    print("INPUT DATA :", data)
    print("MODEL PRED :", prediction)
    print("===================================")

    # Final numeric result
    result_value = int(prediction[0])

    return render_template("result.html", result=result_value)

if __name__ == "__main__":
    app.run(debug=True)
