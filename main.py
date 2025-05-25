from model import train, predict, accuracy
from utils import load_heart_data  # or whatever file your load_heart_data is in

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_heart_data()
input_dim = X_train.shape[1]

# Train the model
model = train(X_train, y_train, input_dim=input_dim, hidden_dim=64, output_dim=1, epochs=3000, lr=0.009)

# Predict on train and test sets
y_train_pred = predict(X_train, model)
y_test_pred = predict(X_test, model)




# Evaluate and print accuracy
print("✅ Train Accuracy:", accuracy(y_train, y_train_pred))
print("✅ Test Accuracy:", accuracy(y_test, y_test_pred))


import numpy as np
import pandas as pd

# Flatten predictions and labels
y_true = y_test.flatten()
y_pred = y_test_pred.flatten()

df_results = pd.DataFrame({
    "Actual": y_true,
    "Predicted": y_pred,
    "Correct": y_true == y_pred
})


print("✅ Correct Predictions:")
print(df_results[df_results["Correct"] == True].head())


print("\n❌ Incorrect Predictions:")
print(df_results[df_results["Correct"] == False].head())


import matplotlib.pyplot as plt

# Count of correct/incorrect
df_results["Correct"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Prediction Accuracy Breakdown")
plt.xticks([0, 1], ["Correct", "Incorrect"], rotation=0)
plt.ylabel("Count")
plt.show()
