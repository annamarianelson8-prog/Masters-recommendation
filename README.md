# KNN Master's Program Recommendation System

Data Warehousing & Data Mining practical project: recommend suitable postgraduate programs by comparing a student's profile with historical students using **K-Nearest Neighbors (KNN)**.

## Features

- Student form: CGPA, UG branch, skills, projects, internships, area of interest, career goals
- Star-schema **data warehouse** (fact + dimension CSVs) with ETL into a clean analytical table
- **KNN** with Euclidean distance and similarity-weighted program voting
- Explanations: shared skills, CGPA closeness, matching interest/goal, project/internship similarity
- Streamlit web UI with warehouse and algorithm explanation tabs

## Project structure

```
DM THEORY PR/
├── app.py                      # Streamlit web application
├── etl.py                      # Join fact + dimensions → clean dataset
├── requirements.txt
├── README.md
├── data/
│   ├── generate_dataset.py     # Synthetic warehouse generator
│   ├── dim_program.csv
│   ├── dim_skill.csv
│   ├── dim_interest.csv
│   ├── fact_student.csv
│   └── student_profiles_clean.csv
└── src/
    ├── preprocess.py           # Encoding + Min-Max scaling
    └── recommender.py          # KNN + explanations
```

## Setup & run

```bash
cd "DM THEORY PR"
pip install -r requirements.txt
python data/generate_dataset.py
python etl.py
streamlit run app.py
```

If the clean CSV is missing, the app will attempt to generate data and run ETL automatically on first launch.

## Dataset (data warehouse)

| Table | Role |
|-------|------|
| `fact_student` | Measures / facts: CGPA, projects, internships, skills, foreign keys |
| `dim_program` | Program name, field, duration, example universities, region |
| `dim_skill` | Skill catalog |
| `dim_interest` | Areas of interest |

~140 synthetic historical students covering programs such as M.Tech CSE, M.Tech AI, M.Sc Data Science, MS CS, MBA Analytics, Cybersecurity, HCI, etc.

## Algorithm notes (for report / viva)

1. **Preprocessing**
   - Numeric: `cgpa`, `projects_count`, `internships_count` → Min-Max scaled to [0, 1]
   - Categorical: `ug_branch`, `area_of_interest`, `career_goal` → one-hot encoding
   - Multi-label skills → binary columns (Python, Java, SQL, ML, …)

2. **KNN**
   - Distance: Euclidean on the scaled feature vector
   - *k* selectable in the UI (3, 5, or 7)
   - Each of the *k* neighbors votes for their chosen Master's program with weight `1 / (distance + ε)`

3. **Recommendation**
   - Programs ranked by total similarity weight
   - Confidence % = program weight / sum of all neighbor weights
   - Contributing factors derived by comparing the query student to neighbors who chose that program

4. **Why this fits Data Warehousing + Data Mining**
   - Warehouse: star schema + ETL producing an analysis-ready table
   - Mining: instance-based classification / recommendation via KNN on that table

## Demo tips

Try two contrasting profiles:

1. **ML-heavy** — CGPA 8.8, skills Python/ML/Deep Learning, interest AI, goal ML Engineer → expect M.Tech AI / M.Sc or MS Data Science
2. **Web / product** — CGPA 7.4, skills Web/UI-UX/Java, interest Product & UX, goal UX Designer → expect M.Des HCI or related software tracks

Change *k* and region/category filters in the sidebar to show how recommendations shift.
