from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])
        region = int(request.form["region"])

        # Prepare input features
        features = np.array([[age, sex, bmi, children, smoker, region]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Round result
        output = round(prediction, 2)

        return render_template(
            "index.html",
            prediction_text=f"Estimated Insurance Cost: ₹ {output}"
        )

    except Exception as e:
        # Error handling
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

# Run server
if __name__ == "__main__":
    app.run(debug=True)
