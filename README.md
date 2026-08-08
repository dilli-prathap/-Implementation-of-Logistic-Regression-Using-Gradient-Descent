# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Import Required Libraries

Load and View the Dataset

Drop Unnecessary Columns

Convert Categorical Columns to category Type

Convert Categories to Numeric Codes

Separate Features (X) and Target (y)

Initialize Model Parameters (Weights)

Define the Sigmoid Function

Define the Loss (Cost) Function

Implement Gradient Descent

Define Prediction Function

Make Predictions and Compute Accuracy

Predict for New Students

## Program:
```PYTHON
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DILLI PRATHAP
Register Number:  212224110014
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Dataset
data = pd.read_csv('Placement_Data.csv')

print("Original Data:")
print(data.head())

# 2. Drop Unnecessary Columns
data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)

print("\nAfter dropping 'sl_no' and 'salary':")
print(data.head())

# 3. Convert Categorical Columns to 'category' Type
data["gender"] = data["gender"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')
data["hsc_s"] = data["hsc_s"].astype('category')

print("\nData types after converting to 'category':")
print(data.dtypes)

# 4. Convert Categories to Numeric Codes
data["gender"] = data["gender"].cat.codes
data["ssc_b"] = data["ssc_b"].cat.codes
data["hsc_b"] = data["hsc_b"].cat.codes
data["degree_t"] = data["degree_t"].cat.codes
data["workex"] = data["workex"].cat.codes
data["specialisation"] = data["specialisation"].cat.codes
data["status"] = data["status"].cat.codes
data["hsc_s"] = data["hsc_s"].cat.codes

print("\nData after converting categories to numeric codes:")
print(data.head())

# 5. Separate Features (X) and Target (y)
x = data.iloc[:, :-1].values   # all columns except last
y = data.iloc[:, -1].values    # last column (status)

print("\nFeature matrix X shape:", x.shape)
print("Target vector y shape:", y.shape)

# 6. Initialize Parameters
theta = np.random.randn(x.shape[1])  # Random initial weights for each feature

print("\nInitial theta (weights):")
print(theta)

# 7. Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 8. Define Loss Function (optional to print/check)
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
    # Added small 1e-10 to avoid log(0)

# 9. Implement Gradient Descent
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient

        # Optional: print loss every 100 iterations
        if (i + 1) % 100 == 0:
            current_loss = loss(theta, X, y)
            print(f"Iteration {i+1}, Loss: {current_loss:.4f}")

    return theta

# 10. Train the Model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

print("\nFinal theta (weights) after training:")
print(theta)

# 11. Define Prediction Function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# 12. Make Predictions and Compute Accuracy
y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)

print("\nTraining Accuracy:", accuracy)
print("Predicted labels (first 20):")
print(y_pred[:20])

# 13. Predict for New Students
# NOTE: The order of features must match 'x' columns exactly.
# Example new students (values must match the encoded format used above)
xnew1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
xnew2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])

y_prednew1 = predict(theta, xnew1)
y_prednew2 = predict(theta, xnew2)

print("\nPrediction for new student 1 (0 = Not Placed, 1 = Placed):", y_prednew1[0])
print("Prediction for new student 2 (0 = Not Placed, 1 = Placed):", y_prednew2[0])
```

## Output:
<img width="932" height="641" alt="632722677-3550980f-d4a0-4cfa-9e51-3c1b8f48490d" src="https://github.com/user-attachments/assets/ef0b88bc-02f8-49aa-b5ba-a79376a1a165" />
<img width="815" height="722" alt="632722858-f39a3a9f-5543-4022-9176-c17129f49f5f" src="https://github.com/user-attachments/assets/26fc6167-8768-49ec-8c7c-0733457f0e70" />
<img width="861" height="527" alt="632722957-93933408-c7be-47d8-892b-a84a94d542cb" src="https://github.com/user-attachments/assets/90619177-7d74-4fac-b8b3-db6591cc6ec7" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

