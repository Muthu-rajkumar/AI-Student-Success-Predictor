"""
train.py
--------
This file trains machine learning models to predict a student's final_marks
based on their habits (attendance, study hours, sleep, etc.)

Steps (beginner-friendly, one at a time):
1. Load the dataset (data/student_data.csv).
2. Split it into "inputs" (X = the features) and "output" (y = final_marks).
3. Turn the "department" text column into numbers (models only understand numbers).
4. Split data into a training set (to learn from) and a test set (to check accuracy).
5. Train THREE different models: Linear Regression, Decision Tree, Random Forest.
6. Compare their accuracy so you can show a comparison table (great for your report).
7. Save the BEST model to models/predictor.pkl so app.py can use it later.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -----------------------------------------------------------------------
# Step 1: Load the dataset
# -----------------------------------------------------------------------
student_data = pd.read_csv("data/student_data.csv")

# -----------------------------------------------------------------------
# Step 2: Turn "department" (text like "CSE", "IT") into numeric columns.
# This is called "one-hot encoding". Example:
# department = "CSE"  ->  department_CSE = 1, department_ECE = 0, ...
# We save the exact column names used here, because app.py must build
# the SAME columns later when a user enters a new student's data.
# -----------------------------------------------------------------------
student_data_encoded = pd.get_dummies(student_data, columns=["department"])

# -----------------------------------------------------------------------
# Step 3: Separate the table into X (inputs/features) and y (what we predict)
# -----------------------------------------------------------------------
target_column = "final_marks"
feature_columns = [col for col in student_data_encoded.columns if col != target_column]

X = student_data_encoded[feature_columns]
y = student_data_encoded[target_column]

# -----------------------------------------------------------------------
# Step 4: Split into training data (80%) and testing data (20%).
# The model learns from the training data, and we check its accuracy
# on the testing data (data it has NEVER seen before).
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------------------
# Step 5: Define the three models we want to compare.
# We store them in a dictionary so we can loop through them easily.
# -----------------------------------------------------------------------
models_to_try = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
}

# This list will store the results (name, accuracy, error) for each model.
results = []

best_model = None
best_model_name = None
best_r2_score = -999  # start very low so the first model always beats it

# -----------------------------------------------------------------------
# Step 6: Train each model, test it, and record how well it did.
# -----------------------------------------------------------------------
for model_name, model in models_to_try.items():

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Ask the model to predict marks for the test students
    predictions = model.predict(X_test)

    # r2_score tells us how much of the variation in marks the model explains.
    # 1.0 = perfect prediction, 0.0 = no better than guessing the average.
    r2 = r2_score(y_test, predictions)

    # mean_absolute_error tells us, on average, how many marks the
    # prediction was off by (easier for students to understand than r2).
    mae = mean_absolute_error(y_test, predictions)

    results.append({"Model": model_name, "R2 Score": round(r2, 3), "Avg Error (marks)": round(mae, 2)})

    print(f"{model_name}: R2 Score = {r2:.3f}, Average Error = {mae:.2f} marks")

    # Keep track of whichever model scored the highest R2 so far.
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model = model
        best_model_name = model_name

# -----------------------------------------------------------------------
# Step 7: Show a clean comparison table (this is great to screenshot for
# your project report / resume).
# -----------------------------------------------------------------------
comparison_table = pd.DataFrame(results)
print("\n=== Model Comparison Table ===")
print(comparison_table.to_string(index=False))
print(f"\nBest model: {best_model_name} (R2 = {best_r2_score:.3f})")

# -----------------------------------------------------------------------
# Step 8: Save the best model + the list of feature column names.
# We save the feature column names too, because app.py needs to know the
# EXACT same column order when making new predictions.
# -----------------------------------------------------------------------
joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "comparison_table": comparison_table,
    },
    "models/predictor.pkl",
)

print("\nSaved best model to models/predictor.pkl")
