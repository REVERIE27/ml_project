from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# -------- DATA --------
X = np.array([
    [1, 7, 50],
    [2, 6, 55],
    [3, 8, 65],
    [4, 5, 60],
    [5, 7, 70],
    [6, 6, 75],
    [7, 8, 85],
    [8, 5, 80]
], dtype=float)

y = np.array([52, 57, 66, 63, 72, 78, 88, 84], dtype=float)

# -------- NORMALIZATION --------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X - X_mean) / X_std

# -------- TRAIN MODEL --------
weights = np.zeros(3)
bias = 0

lr = 0.01
epochs = 2000
n = len(X)

for i in range(epochs):
    y_pred = np.dot(X, weights) + bias

    dw = (-2/n) * np.dot(X.T, (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    weights -= lr * dw
    bias -= lr * db

# -------- ROUTES --------


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            hours = float(request.form.get("hours"))
            sleep = float(request.form.get("sleep"))
            previous = float(request.form.get("previous"))

            # Normalize input SAME as training
            features = np.array([hours, sleep, previous], dtype=float)
            features = (features - X_mean) / X_std

            prediction = round(np.dot(features, weights) + bias, 2)

            print("Input:", features)
            print("Prediction:", prediction)

        except Exception as e:
            print("Error:", e)
            error = "Invalid input."

    return render_template("index.html", prediction=prediction, error=error)


# -------- RUN --------
if __name__ == "__main__":
    app.run(debug=True)
