from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/credit_card_fraud_detection_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]

        if prediction == 1:
            result_text = f"🚨 Fraudulent Transaction Detected! (Probability: {probability:.2%})"
        else:
            result_text = f"✅ Legitimate Transaction (Probability of Fraud: {probability:.2%})"

        return render_template("index.html", prediction_text=result_text)
    except:
        return render_template("index.html", prediction_text="⚠️ Please enter valid numeric values for all fields.")

if __name__ == "__main__":
    app.run(debug=True)
