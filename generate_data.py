"""
generate_data.py
-----------------
This file creates a FAKE (synthetic) student dataset so you can train and test
your models without needing real college data first.

Later, you can replace this fake data with a real CSV collected from your
department (attendance register, marks sheet, a Google Form for sleep/social
media habits, etc). As long as the real CSV has the SAME column names used
here, the rest of the project (train.py, app.py) will work without changes.

We use simple, beginner-friendly steps:
1. Decide how many fake students we want.
2. For each student, randomly generate realistic values for every feature.
3. Make the "final_marks" column depend on those features in a realistic way
   (so the ML model actually has real patterns to learn, not pure randomness).
4. Save everything into data/student_data.csv
"""

import numpy as np
import pandas as pd

# Step 1: Set a "seed" so that every time we run this file, we get the SAME
# random numbers. This makes our project reproducible (important for demos).
np.random.seed(42)

# Step 2: Decide how many fake students to create.
number_of_students = 500

# Step 3: Create a list of departments to randomly assign to students.
department_list = ["AI&DS", "CSE", "ECE", "IT"]

# Step 4: Generate each feature column one by one using numpy's random tools.

# attendance_percent: most students are between 40% and 100%
attendance_percent = np.random.uniform(40, 100, number_of_students)

# daily_study_hours: most students study between 0 and 6 hours a day
daily_study_hours = np.random.uniform(0, 6, number_of_students)

# weekly_consistency_score: how consistent their study routine is (0 to 10)
# 10 = studies almost the same amount every day, 0 = very irregular
weekly_consistency_score = np.random.uniform(0, 10, number_of_students)

# previous_marks: their marks in the last exam (0 to 100)
previous_marks = np.random.uniform(35, 95, number_of_students)

# assignments_submitted: out of 10 assignments given in the semester
assignments_submitted = np.random.randint(0, 11, number_of_students)
assignment_completion_rate = (assignments_submitted / 10) * 100

# sleep_hours: average sleep per night (3 to 10 hours)
sleep_hours = np.random.uniform(3, 10, number_of_students)

# social_media_hours: hours spent per day on Instagram/YouTube etc.
social_media_hours = np.random.uniform(0, 8, number_of_students)

# department: randomly assign one of the 4 departments to each student
department = np.random.choice(department_list, number_of_students)

# Step 5: Now create "final_marks" using a formula that mimics real life:
# - Higher attendance -> higher marks
# - More study hours -> higher marks
# - Higher previous marks -> higher marks (past performance predicts future)
# - More assignments submitted -> higher marks
# - Enough sleep (7-9 hrs is ideal) -> higher marks, too little/too much hurts
# - More social media time -> LOWER marks
# - We also add some random "noise" because real life is never a perfect formula

# sleep_effect: we reward students who sleep close to 8 hours,
# and penalize students who sleep too little or too much.
sleep_effect = -1.5 * np.abs(sleep_hours - 8)

final_marks = (
    0.30 * attendance_percent
    + 4.0 * daily_study_hours
    + 0.25 * previous_marks
    + 0.15 * assignment_completion_rate
    + sleep_effect
    - 2.0 * social_media_hours
    + 1.0 * weekly_consistency_score
    + np.random.normal(0, 5, number_of_students)  # random noise
)

# Step 6: Clip marks so they stay within a realistic 0-100 range.
final_marks = np.clip(final_marks, 0, 100)

# Step 7: Put all the columns together into one table (DataFrame).
student_data = pd.DataFrame({
    "attendance_percent": attendance_percent.round(1),
    "daily_study_hours": daily_study_hours.round(2),
    "weekly_consistency_score": weekly_consistency_score.round(1),
    "previous_marks": previous_marks.round(1),
    "assignments_submitted": assignments_submitted,
    "assignment_completion_rate": assignment_completion_rate.round(1),
    "sleep_hours": sleep_hours.round(1),
    "social_media_hours": social_media_hours.round(1),
    "department": department,
    "final_marks": final_marks.round(1),
})

# Step 8: Save the table as a CSV file inside the data/ folder.
student_data.to_csv("data/student_data.csv", index=False)

print("Done! Fake dataset created at data/student_data.csv")
print(student_data.head())
