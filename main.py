import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: Create Dataset
# -------------------------------
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8])
final_score = np.array([52, 57, 66, 63, 72, 78, 88, 84])

# -------------------------------
# STEP 2: Initialize parameters
# -------------------------------
m = 0   # slope
b = 0   # intercept
learning_rate = 0.01
epochs = 1000

n = len(hours_studied)

# -------------------------------
# STEP 3: Train Model (Gradient Descent)
# -------------------------------
for i in range(epochs):
    y_pred = m * hours_studied + b

    # Calculate gradients
    dm = (-2/n) * sum(hours_studied * (final_score - y_pred))
    db = (-2/n) * sum(final_score - y_pred)

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# -------------------------------
# STEP 4: Print learned values
# -------------------------------
print("Slope (m):", m)
print("Intercept (b):", b)

# -------------------------------
# STEP 5: Make Prediction
# -------------------------------
study_input = 5
predicted_score = m * study_input + b

print("\nPredicted score for 5 hours study:", predicted_score)

# -------------------------------
# STEP 6: Plot Graph
# -------------------------------
plt.scatter(hours_studied, final_score, label="Actual Data")
plt.plot(hours_studied, m * hours_studied + b, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.title("Linear Regression From Scratch")
plt.legend()
plt.show()
