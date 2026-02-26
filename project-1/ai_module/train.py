import pandas as pd
import pickle 
import os
from regression_model import train_model , evaluation_model , tune_model

df = pd.read_csv("../data/student_data.csv")

X = df[["Hours"]]
y = df["Marks"]

#train
model = train_model(X, y, degree=3, alpha=1)
# model, best_params, best_score = tune_model(X, y)

os.makedirs("../models",exist_ok=True)

with open("../models/model.pkl","wb") as f:
    pickle.dump(model,f)

# Evaluate
results = evaluation_model(model , X , y)

print("best parameter:",best_params)
print("best score:",best_score)
print("Fold Scores:", results["scores"])
print("Mean R2:", results["mean_r2"])
print("Std R2:", results["std_r2"])

print("model train successfully.")