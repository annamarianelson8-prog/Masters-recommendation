"""
Streamlit web app: KNN-based Master's Program Recommendation System.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import SKILL_COLUMNS
from src.recommender import MastersRecommender

CLEAN_PATH = ROOT / "data" / "student_profiles_clean.csv"

UG_BRANCHES = ["CSE", "IT", "ECE", "EEE", "Mechanical", "Mathematics", "Statistics"]
INTERESTS = [
    "Artificial Intelligence",
    "Data Science & Analytics",
    "Software Development",
    "Cybersecurity",
    "Product & UX Design",
    "Cloud & DevOps",
    "Business & Strategy",
]
CAREER_GOALS = [
    "Research Scientist",
    "Data Scientist",
    "Software Engineer",
    "ML Engineer",
    "Security Analyst",
    "Product Manager",
    "UX Designer",
    "Cloud Architect",
    "Business Analyst",
]


@st.cache_resource
def load_recommender() -> MastersRecommender:
    return MastersRecommender()


def ensure_data() -> None:
    if CLEAN_PATH.exists():
        return
    # Auto-bootstrap warehouse if missing
    from data.generate_dataset import generate
    from etl import run_etl

    generate()
    run_etl()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main-title { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.2rem; }
        .subtitle { color: #5a6570; margin-bottom: 1.2rem; }
        .rec-card {
            border: 1px solid #d9e2ec;
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        }
        .rec-rank { font-size: 0.8rem; color: #486581; text-transform: uppercase; letter-spacing: 0.04em; }
        .rec-name { font-size: 1.15rem; font-weight: 650; color: #102a43; margin: 0.15rem 0; }
        .rec-meta { color: #486581; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(rank: int, rec: dict) -> None:
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-rank">Recommendation #{rank}</div>
            <div class="rec-name">{rec['program']}</div>
            <div class="rec-meta">
                {rec.get('field', '')} · {rec.get('duration_years', '')} years ·
                {rec.get('region', '')} · Confidence {rec.get('confidence_pct', 0)}%
            </div>
            <div class="rec-meta">Examples: {rec.get('example_universities', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Master's Recommendation System",
        page_icon="🎓",
        layout="wide",
    )
    inject_styles()
    ensure_data()

    st.markdown(
        '<div class="main-title">Master\'s Program Recommendation System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">KNN-based recommendations from a student data warehouse — '
        "CGPA, skills, projects, internships, interests & career goals.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("KNN Settings")
        k = st.select_slider("Number of neighbors (k)", options=[3, 5, 7], value=5)
        region = st.selectbox("Region filter", ["All", "India", "Abroad"])
        category = st.selectbox(
            "Category filter", ["All", "Technical", "Management", "Design"]
        )
        st.caption("Filters apply after finding neighbors, then re-rank programs.")

    tab_rec, tab_wh, tab_knn = st.tabs(
        ["Get Recommendations", "Data Warehouse", "How KNN Works"]
    )

    with tab_rec:
        st.subheader("Your profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.2, step=0.1)
            ug_branch = st.selectbox("UG Branch", UG_BRANCHES)
        with c2:
            projects = st.number_input("Projects completed", min_value=0, max_value=15, value=4)
            internships = st.number_input("Internships", min_value=0, max_value=8, value=2)
        with c3:
            interest = st.selectbox("Area of interest", INTERESTS)
            career_goal = st.selectbox("Career goal", CAREER_GOALS)

        skills = st.multiselect(
            "Technical skills",
            options=SKILL_COLUMNS,
            default=["Python", "SQL", "Machine Learning"],
        )

        run = st.button("Recommend Master's Programs", type="primary")

        if run:
            if not skills:
                st.warning("Please select at least one technical skill.")
            else:
                student = {
                    "cgpa": cgpa,
                    "ug_branch": ug_branch,
                    "projects_count": int(projects),
                    "internships_count": int(internships),
                    "area_of_interest": interest,
                    "career_goal": career_goal,
                    "skills": skills,
                }
                try:
                    recommender = load_recommender()
                    result = recommender.recommend(
                        student,
                        k=k,
                        region_filter=region,
                        category_filter=category,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not generate recommendations: {exc}")
                    st.stop()

                st.subheader("Top recommended programs")
                if not result.recommendations:
                    st.info("No recommendations found for the current filters.")
                else:
                    for i, rec in enumerate(result.recommendations, start=1):
                        render_recommendation_card(i, rec)
                        with st.expander(f"Why this match — {rec['program']}"):
                            for factor in result.explanations.get(rec["program"], []):
                                st.markdown(f"- {factor}")

                st.subheader("Similar historical students (neighbors)")
                st.dataframe(result.neighbors, use_container_width=True, hide_index=True)

    with tab_wh:
        st.subheader("Star-schema data warehouse")
        st.markdown(
            """
            Historical student records are stored as a small **star schema** extract:

            - **Fact table** `fact_student.csv` — one row per past student (CGPA, projects,
              internships, skills, chosen program key)
            - **Dimensions**
              - `dim_program.csv` — program name, field, duration, universities, region
              - `dim_skill.csv` — skill catalog
              - `dim_interest.csv` — areas of interest
            - **ETL** (`etl.py`) joins facts + dimensions into `student_profiles_clean.csv`
              used by KNN.
            """
        )
        st.markdown(
            """
            ```
                      ┌─────────────────┐
                      │  dim_program    │
                      └────────┬────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            │           fact_student              │
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
           dim_interest                 dim_skill
            ```
            """
        )
        if CLEAN_PATH.exists():
            clean = pd.read_csv(CLEAN_PATH)
            st.metric("Historical student profiles", len(clean))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sample fact extract (clean)**")
                st.dataframe(
                    clean[
                        [
                            "student_id",
                            "cgpa",
                            "ug_branch",
                            "skills",
                            "chosen_masters_program",
                        ]
                    ].head(8),
                    use_container_width=True,
                    hide_index=True,
                )
            with c2:
                st.markdown("**Programs in warehouse**")
                prog = (
                    clean.groupby("chosen_masters_program")
                    .size()
                    .reset_index(name="students")
                    .sort_values("students", ascending=False)
                )
                st.dataframe(prog, use_container_width=True, hide_index=True)

    with tab_knn:
        st.subheader("How the recommender works")
        st.markdown(
            """
            1. **Preprocessing** — Numeric features (CGPA, projects, internships) are
               Min-Max scaled; branch, interest, and career goal are one-hot encoded;
               skills become binary columns.
            2. **Distance** — Euclidean distance on the scaled feature vector finds the
               *k* nearest historical students.
            3. **Weighted vote** — Each neighbor votes for the Master's program they
               chose; closer neighbors get higher weight (`1 / (distance + ε)`).
            4. **Explanation** — For each top program, the app lists contributing factors
               (CGPA closeness, shared skills, matching interest/goal, project/internship
               similarity).

            This is a classic **instance-based / lazy learning** data-mining technique:
            no separate training of a parametric model — recommendations come directly
            from similar past cases stored in the warehouse.
            """
        )


if __name__ == "__main__":
    main()
