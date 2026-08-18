# AI-Powered Student Success Predictor

A student performance predictor upgraded with risk detection, an AI "what-if"
recommendation engine, a rule-based chatbot, and an interactive Streamlit
dashboard — built to go beyond the typical "attendance + hours -> marks"
college project.

## Features

- Predicts final marks from attendance, study hours, sleep, assignments,
  social media usage, and weekly consistency
- Compares 3 ML models (Linear Regression, Decision Tree, Random Forest)
  and automatically picks the best one
- Risk detection: Low / Medium / High risk labels
- AI recommendation engine: tests realistic "what-if" changes (more
  attendance, more study time, less social media, better sleep, more
  assignments) and reports which ones improve the score the most
- Simple rule-based chatbot that answers using the same recommendation logic
- Department comparison (AI&DS / CSE / ECE / IT)
- Bulk CSV upload to predict an entire class at once, with downloadable results

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the (synthetic) dataset
python generate_data.py

# 2. Train and compare models — saves the best one to models/predictor.pkl
python train.py

# 3. Launch the dashboard
streamlit run app.py
```

## Using Real Data Instead of Synthetic Data

Replace `data/student_data.csv` with your own CSV using the same column
names, then re-run `python train.py`. Good sources: attendance register
export, marks sheet, and a short Google Form asking students about daily
study hours, sleep, and social media usage (with consent, kept anonymous).

## Project Architecture

```
Student Data
      ↓
Data Cleaning / Feature Engineering  (generate_data.py / train.py)
      ↓
ML Model Training + Comparison        (train.py)
      ↓
Prediction                            (recommendation.py)
      ↓
AI Recommendation Engine              (recommendation.py)
      ↓
Dashboard + Chatbot                   (app.py)
```

## Repository Structure

```
AI-Student-Success-Predictor/
│
├── data/
│   └── student_data.csv
├── models/
│   └── predictor.pkl
├── notebooks/
│   └── training.ipynb        (optional — for showing your work-in-progress)
├── app.py                    (Streamlit dashboard)
├── train.py                  (model training + comparison)
├── generate_data.py          (synthetic dataset generator)
├── recommendation.py         (AI recommendation engine + chatbot logic)
├── requirements.txt
├── README.md
└── screenshots/
```

## Roadmap / Ideas Not Yet Built

These are worth adding once the core is working and demoed, rather than
all at once:
- Login system (Streamlit doesn't have this built in — needs
  `streamlit-authenticator` or a small database)
- PDF report generation (`reportlab` or `fpdf2`)
- Email report (`smtplib`)
- XGBoost as a 4th model to compare (`pip install xgboost`)

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/AI-Student-Success-Predictor.git
git branch -M main
git push -u origin main
```
