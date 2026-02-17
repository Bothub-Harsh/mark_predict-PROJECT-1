import pickle
import numpy as np

# Load model
with open("./models/model.pkl", "rb") as f:
    model = pickle.load(f)

hours = float(input("Enter study hours: "))
prediction = model.predict(np.array([[hours]]))

print(f"Predicted Marks: {prediction[0]}")
