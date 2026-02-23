# Train model
model = LinearRegression()
model.fit(X, y)

# Slope tells how much output changes when input increases by 1
# Intercept = value when input = 0

Inside model.fit(X, y)

Step 1 — Read data
X = Hours studied
y = Marks obtained


Model sees points like:

(1,20) (2,30) (3,45) (5,70)

Step 2 — Try many lines
Computer imagines different equations:
Marks = 5×Hours + 10   ❌ bad fit
Marks = 20×Hours - 5   ❌ bad fit
Marks = 12×Hours + 6   👍 better
Marks = 12.5×Hours +5  ⭐ best

Step 3 — Measure error

For every line it calculates mistake:

Example:

Real marks = 45
Predicted = 40
Error = 25

Goal:

Find line with minimum total error

This is called:

Least Squares Method

Step 4 — Choose best line

Finally sklearn stores:

model.coef_      # slope (m)
model.intercept_ # intercept (c)


Your model is now trained

Step 5 — Ready for prediction

Now:

model.predict([[6]])


No learning happens now ❌
Only calculation happens ✔️

---------------

fit() = learning phase 🧠
predict() = using knowledge 📊

pickle converts Python object → bytes → saves to file

LinearRegression object
(with slope & intercept learned)
↓
converted to binary
↓
stored in model.pkl
