"""
recommendation.py
------------------
This file contains the "smart" logic that makes this project stand out:

1. predict_marks_for_student()  -> uses the trained model to predict marks
2. get_risk_level()              -> turns a predicted score into High/Medium/Low risk
3. generate_recommendations()    -> tries small realistic changes (like "+1 study hour")
                                     and tells the student how much their score could improve
4. chatbot_response()            -> a simple rule-based chatbot that answers common questions
                                     using the SAME recommendation logic (no external AI API needed)

Keeping this logic in its own file (separate from app.py) means:
- app.py stays focused on the DASHBOARD/UI
- this file stays focused on the "AI brain"
- you can re-use or test this logic without running the whole Streamlit app
"""

import pandas as pd


# -----------------------------------------------------------------------
# Step 1: Turn one student's raw inputs into the exact table format
# the model expects (same columns, same order, department one-hot encoded).
# -----------------------------------------------------------------------
def build_feature_row(student_inputs, feature_columns):
    """
    student_inputs: a dictionary like:
        {
            "attendance_percent": 75,
            "daily_study_hours": 2,
            "weekly_consistency_score": 5,
            "previous_marks": 70,
            "assignments_submitted": 7,
            "assignment_completion_rate": 70,
            "sleep_hours": 6,
            "social_media_hours": 3,
            "department": "AI&DS",
        }
    feature_columns: the list of column names the model was trained on
                      (saved inside models/predictor.pkl)

    Returns: a single-row pandas DataFrame ready to feed into model.predict()
    """
    # Start with a dictionary of all zeros for every column the model expects.
    row = {col: 0 for col in feature_columns}

    # Fill in the normal numeric features.
    for key in [
        "attendance_percent", "daily_study_hours", "weekly_consistency_score",
        "previous_marks", "assignments_submitted", "assignment_completion_rate",
        "sleep_hours", "social_media_hours",
    ]:
        if key in row:
            row[key] = student_inputs[key]

    # Handle the department one-hot column, e.g. "department_CSE" = 1
    department_column_name = f"department_{student_inputs['department']}"
    if department_column_name in row:
        row[department_column_name] = 1

    return pd.DataFrame([row])[feature_columns]


# -----------------------------------------------------------------------
# Step 2: Predict marks for one student.
# -----------------------------------------------------------------------
def predict_marks_for_student(model, feature_columns, student_inputs):
    feature_row = build_feature_row(student_inputs, feature_columns)
    predicted_marks = model.predict(feature_row)[0]

    # Keep the prediction within a realistic 0-100 range.
    predicted_marks = max(0, min(100, predicted_marks))
    return round(predicted_marks, 1)


# -----------------------------------------------------------------------
# Step 3: Convert a predicted score into a simple risk label.
# You can tune these thresholds to match your college's pass/fail rules.
# -----------------------------------------------------------------------
def get_risk_level(predicted_marks):
    if predicted_marks < 50:
        return "High Risk"
    elif predicted_marks < 70:
        return "Medium Risk"
    else:
        return "Low Risk"


# -----------------------------------------------------------------------
# Step 4: The "what-if" recommendation engine.
# Idea: try small, realistic improvements one at a time (more attendance,
# more study hours, more sleep, less social media) and see how much each
# one would raise the predicted score. Then suggest the ones that help most.
# -----------------------------------------------------------------------
def generate_recommendations(model, feature_columns, student_inputs, top_n=3):
    baseline_prediction = predict_marks_for_student(model, feature_columns, student_inputs)

    # Each "what-if" scenario: (readable description, how to change the inputs)
    candidate_changes = []

    # Scenario 1: increase attendance by 10 points (capped at 100)
    if student_inputs["attendance_percent"] < 95:
        new_inputs = dict(student_inputs)
        new_inputs["attendance_percent"] = min(100, student_inputs["attendance_percent"] + 10)
        description = (
            f"Increase attendance from {student_inputs['attendance_percent']}% "
            f"to {new_inputs['attendance_percent']}%"
        )
        candidate_changes.append((description, new_inputs))

    # Scenario 2: study 1 extra hour per day
    if student_inputs["daily_study_hours"] < 6:
        new_inputs = dict(student_inputs)
        new_inputs["daily_study_hours"] = round(student_inputs["daily_study_hours"] + 1, 1)
        description = f"Study 1 extra hour daily (from {student_inputs['daily_study_hours']} to {new_inputs['daily_study_hours']} hrs)"
        candidate_changes.append((description, new_inputs))

    # Scenario 3: reduce social media usage by 1.5 hours
    if student_inputs["social_media_hours"] > 1:
        new_inputs = dict(student_inputs)
        new_inputs["social_media_hours"] = max(0, round(student_inputs["social_media_hours"] - 1.5, 1))
        description = f"Cut social media time by 1.5 hrs (from {student_inputs['social_media_hours']} to {new_inputs['social_media_hours']} hrs)"
        candidate_changes.append((description, new_inputs))

    # Scenario 4: move sleep closer to the healthy 8-hour mark
    if abs(student_inputs["sleep_hours"] - 8) > 1:
        new_inputs = dict(student_inputs)
        step = 1 if student_inputs["sleep_hours"] < 8 else -1
        new_inputs["sleep_hours"] = round(student_inputs["sleep_hours"] + step, 1)
        description = f"Adjust sleep toward 8 hrs (from {student_inputs['sleep_hours']} to {new_inputs['sleep_hours']} hrs)"
        candidate_changes.append((description, new_inputs))

    # Scenario 5: submit more assignments (if not already at 10/10)
    if student_inputs["assignments_submitted"] < 10:
        new_inputs = dict(student_inputs)
        new_inputs["assignments_submitted"] = min(10, student_inputs["assignments_submitted"] + 2)
        new_inputs["assignment_completion_rate"] = (new_inputs["assignments_submitted"] / 10) * 100
        description = f"Submit 2 more assignments ({student_inputs['assignments_submitted']} -> {new_inputs['assignments_submitted']} out of 10)"
        candidate_changes.append((description, new_inputs))

    # Now test each scenario and measure how much it improves the score.
    scored_recommendations = []
    for description, new_inputs in candidate_changes:
        new_prediction = predict_marks_for_student(model, feature_columns, new_inputs)
        improvement = round(new_prediction - baseline_prediction, 1)
        if improvement > 0:
            scored_recommendations.append({
                "action": description,
                "new_predicted_marks": new_prediction,
                "improvement": improvement,
            })

    # Sort so the BIGGEST improvement is shown first.
    scored_recommendations.sort(key=lambda item: item["improvement"], reverse=True)

    return baseline_prediction, scored_recommendations[:top_n]


# -----------------------------------------------------------------------
# Step 5: A simple rule-based chatbot.
# It looks for keywords in the student's question and reuses the SAME
# recommendation engine above to answer with real numbers, not generic advice.
# -----------------------------------------------------------------------
def chatbot_response(user_question, model, feature_columns, student_inputs):
    question = user_question.lower()

    baseline_prediction, recommendations = generate_recommendations(
        model, feature_columns, student_inputs
    )

    # Greeting
    if any(word in question for word in ["hi", "hello", "hey"]):
        return "Hi! Ask me things like 'how can I improve my marks?' or 'am I at risk?'"

    # Risk-related question
    if "risk" in question or "fail" in question:
        risk_level = get_risk_level(baseline_prediction)
        return (
            f"Based on your current habits, your predicted score is {baseline_prediction}% "
            f"which puts you in the **{risk_level}** category."
        )

    # Improvement-related question (the main use case)
    if any(word in question for word in ["improve", "better", "increase", "raise", "help"]):
        if not recommendations:
            return "You're already doing great across attendance, study hours, sleep, and assignments!"

        top = recommendations[0]
        reply = (
            f"Your predicted score is {baseline_prediction}%. "
            f"If you {top['action'].lower()}, your score can improve to "
            f"{top['new_predicted_marks']}% (+{top['improvement']} marks)."
        )
        if len(recommendations) > 1:
            extra = recommendations[1]
            reply += (
                f" Another good option: {extra['action'].lower()}, "
                f"which could push you to {extra['new_predicted_marks']}%."
            )
        return reply

    # Attendance-specific
    if "attendance" in question:
        return f"Your current attendance is {student_inputs['attendance_percent']}%. Aim for 85%+ for the best results."

    # Sleep-specific
    if "sleep" in question:
        return f"You're averaging {student_inputs['sleep_hours']} hrs of sleep. 7-9 hrs is ideal for focus and memory."

    # Default fallback
    return "I can help with questions about your risk level or how to improve your predicted marks — try asking 'how can I improve my score?'"
