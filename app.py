from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("model", "fraud_pipeline.pkl")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the form
        data = {
            "step": float(request.form["step"]),
            "type": request.form["type"],
            "amount": float(request.form["amount"]),
            "oldbalanceOrig": float(request.form["oldbalanceOrg"]),
            "newbalanceOrig": float(request.form["newbalanceOrig"]),
            "oldbalanceDest": float(request.form["oldbalanceDest"]),
            "newbalanceDest": float(request.form["newbalanceDest"])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = ("🚨 Fraudulent Transaction" if prediction == 1
                  else "✅ Legitimate Transaction")

        # Render result page
        return render_template("results.html", result=result, data=data)

    except Exception as e:
        return render_template("results.html",
                               result=f"⚠️ Error: {e}", data={})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
