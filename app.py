"""
app.py
------
This is the Streamlit DASHBOARD. Run it with:

    streamlit run app.py

It gives you:
- A form to enter one student's habits
- A predicted score + risk level
- AI recommendations (what-if suggestions)
- A simple chatbot
- Charts comparing attendance/study hours to marks across the whole dataset
- A department-wise comparison
- CSV upload to predict marks for MANY students at once

NOTE: This app depends on models/predictor.pkl existing.
Run these first, in order:
    python generate_data.py
    python train.py
    streamlit run app.py
"""

import pandas as pd
import joblib
import streamlit as st

from recommendation import (
    predict_marks_for_student,
    get_risk_level,
    generate_recommendations,
    chatbot_response,
)

# -----------------------------------------------------------------------
# Page setup — must be the first Streamlit command in the file.
# -----------------------------------------------------------------------
st.set_page_config(page_title="AI-Powered Student Success Predictor", layout="wide")

# -----------------------------------------------------------------------
# Step 1: Load the trained model (cached so it only loads once, not on
# every click — this keeps the dashboard fast).
# -----------------------------------------------------------------------
@st.cache_resource
def load_model():
    saved = joblib.load("models/predictor.pkl")
    return saved["model"], saved["feature_columns"], saved["model_name"], saved["comparison_table"]


@st.cache_data
def load_dataset():
    return pd.read_csv("data/student_data.csv")


model, feature_columns, model_name, comparison_table = load_model()
full_dataset = load_dataset()

st.title("🎓 AI-Powered Student Success Predictor")
st.caption(f"Currently using: **{model_name}** (best of Linear Regression / Decision Tree / Random Forest)")

# -----------------------------------------------------------------------
# Sidebar: student inputs form
# -----------------------------------------------------------------------
st.sidebar.header("Enter Student Details")

attendance_percent = st.sidebar.slider("Attendance %", 0, 100, 75)
daily_study_hours = st.sidebar.slider("Daily Study Hours", 0.0, 8.0, 2.0, step=0.5)
weekly_consistency_score = st.sidebar.slider("Weekly Consistency Score (0-10)", 0.0, 10.0, 5.0, step=0.5)
previous_marks = st.sidebar.slider("Previous Exam Marks", 0, 100, 65)
assignments_submitted = st.sidebar.slider("Assignments Submitted (out of 10)", 0, 10, 6)
sleep_hours = st.sidebar.slider("Average Sleep Hours", 0.0, 12.0, 6.5, step=0.5)
social_media_hours = st.sidebar.slider("Social Media Hours per Day", 0.0, 10.0, 3.0, step=0.5)
department = st.sidebar.selectbox("Department", ["AI&DS", "CSE", "ECE", "IT"])

student_inputs = {
    "attendance_percent": attendance_percent,
    "daily_study_hours": daily_study_hours,
    "weekly_consistency_score": weekly_consistency_score,
    "previous_marks": previous_marks,
    "assignments_submitted": assignments_submitted,
    "assignment_completion_rate": (assignments_submitted / 10) * 100,
    "sleep_hours": sleep_hours,
    "social_media_hours": social_media_hours,
    "department": department,
}

predict_button = st.sidebar.button("Predict My Score", type="primary")

# -----------------------------------------------------------------------
# Main area: Tabs for different features of the dashboard
# -----------------------------------------------------------------------
tab_predict, tab_charts, tab_department, tab_chatbot, tab_bulk = st.tabs(
    ["📊 Prediction", "📈 Charts", "🏫 Department Comparison", "🤖 Chatbot", "📁 Bulk CSV Upload"]
)

# --- TAB 1: Prediction + Recommendations + Risk -------------------------
with tab_predict:
    if predict_button:
        predicted_marks = predict_marks_for_student(model, feature_columns, student_inputs)
        risk_level = get_risk_level(predicted_marks)

        col1, col2 = st.columns(2)
        col1.metric("Predicted Final Marks", f"{predicted_marks}%")

        risk_color = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
        col2.metric("Risk Level", f"{risk_color[risk_level]} {risk_level}")

        st.subheader("🤖 AI Recommendations")
        baseline, recommendations = generate_recommendations(model, feature_columns, student_inputs)

        if not recommendations:
            st.success("Great job! Your current habits are already well balanced.")
        else:
            for rec in recommendations:
                st.info(
                    f"**{rec['action']}** → predicted score improves to "
                    f"**{rec['new_predicted_marks']}%** (+{rec['improvement']} marks)"
                )
    else:
        st.info("Fill in the details on the left sidebar and click **Predict My Score**.")

# --- TAB 2: Charts -------------------------------------------------------
with tab_charts:
    st.subheader("How Attendance Relates to Marks (whole dataset)")
    st.scatter_chart(full_dataset, x="attendance_percent", y="final_marks")

    st.subheader("How Study Hours Relate to Marks (whole dataset)")
    st.scatter_chart(full_dataset, x="daily_study_hours", y="final_marks")

    st.subheader("How Sleep Relates to Marks (whole dataset)")
    st.scatter_chart(full_dataset, x="sleep_hours", y="final_marks")

    st.subheader("How Social Media Usage Relates to Marks (whole dataset)")
    st.scatter_chart(full_dataset, x="social_media_hours", y="final_marks")

    st.subheader("Model Comparison")
    st.dataframe(comparison_table, use_container_width=True)

# --- TAB 3: Department comparison ----------------------------------------
with tab_department:
    st.subheader("Average Final Marks by Department")
    department_avg = full_dataset.groupby("department")["final_marks"].mean().round(1)
    st.bar_chart(department_avg)

    st.subheader("Average Attendance by Department")
    department_attendance = full_dataset.groupby("department")["attendance_percent"].mean().round(1)
    st.bar_chart(department_attendance)

    st.dataframe(
        full_dataset.groupby("department")[
            ["attendance_percent", "daily_study_hours", "sleep_hours", "final_marks"]
        ].mean().round(1),
        use_container_width=True,
    )

# --- TAB 4: Chatbot -------------------------------------------------------
with tab_chatbot:
    st.subheader("Ask the Assistant")
    st.caption("Try: 'How can I improve my marks?' or 'Am I at risk?' — uses the sidebar values above.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Type your question here")

    if st.button("Ask") and user_question.strip() != "":
        answer = chatbot_response(user_question, model, feature_columns, student_inputs)
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Assistant", answer))

    for speaker, message in reversed(st.session_state.chat_history):
        if speaker == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Assistant:** {message}")

# --- TAB 5: Bulk CSV upload (predict for a whole class at once) ----------
with tab_bulk:
    st.subheader("Upload a CSV to predict marks for many students at once")
    st.caption(
        "CSV must have these columns: attendance_percent, daily_study_hours, "
        "weekly_consistency_score, previous_marks, assignments_submitted, "
        "sleep_hours, social_media_hours, department"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        bulk_data = pd.read_csv(uploaded_file)

        predicted_list = []
        risk_list = []

        # Go through the CSV row by row and predict each student's marks.
        for _, row in bulk_data.iterrows():
            row_inputs = {
                "attendance_percent": row["attendance_percent"],
                "daily_study_hours": row["daily_study_hours"],
                "weekly_consistency_score": row["weekly_consistency_score"],
                "previous_marks": row["previous_marks"],
                "assignments_submitted": row["assignments_submitted"],
                "assignment_completion_rate": (row["assignments_submitted"] / 10) * 100,
                "sleep_hours": row["sleep_hours"],
                "social_media_hours": row["social_media_hours"],
                "department": row["department"],
            }
            pred = predict_marks_for_student(model, feature_columns, row_inputs)
            predicted_list.append(pred)
            risk_list.append(get_risk_level(pred))

        bulk_data["predicted_marks"] = predicted_list
        bulk_data["risk_level"] = risk_list

        st.success(f"Predicted marks for {len(bulk_data)} students.")
        st.dataframe(bulk_data, use_container_width=True)

        # Let the user download the results as a new CSV.
        st.download_button(
            "Download Results as CSV",
            bulk_data.to_csv(index=False).encode("utf-8"),
            "predicted_results.csv",
            "text/csv",
        )
