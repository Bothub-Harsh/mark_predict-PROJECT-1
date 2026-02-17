import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("./data/student_data.csv")

X = df[["Hours"]]
y = df["Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
