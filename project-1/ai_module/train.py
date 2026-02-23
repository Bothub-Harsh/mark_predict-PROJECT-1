import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("../data/student_data.csv")

X = df[["Hours"]]
y = df["Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=101
)

# Train model
model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
    )

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

X_range = np.linspace(X["Hours"].min(), X["Hours"].max(), 100)
X_range = X_range.reshape(-1, 1)

y_range = model.predict(X_range)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mse)
residuals = y_test - predictions    # Residual = actual - predicated

# Blue = real data
# Red = model line
plt.scatter(X, y, color="blue")
plt.plot(X_range, y_range, color="red")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Regression Line")


print("Model Evaluation")
print("----------------")
print("Residuals:", residuals.head())
print("RMSE:", rmse)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

plt.show()




