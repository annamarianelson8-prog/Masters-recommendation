"""
ETL: join fact_student with dimension tables into a clean analytical dataset
for the KNN Master's Recommendation System.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_DIR / "student_profiles_clean.csv"


def run_etl() -> pd.DataFrame:
    fact = pd.read_csv(DATA_DIR / "fact_student.csv")
    programs = pd.read_csv(DATA_DIR / "dim_program.csv")
    interests = pd.read_csv(DATA_DIR / "dim_interest.csv")

    # Star-schema join: fact -> dim_program, dim_interest
    merged = fact.merge(programs, on="program_id", how="left")
    merged = merged.merge(
        interests[["interest_id", "interest_name"]],
        on="interest_id",
        how="left",
        suffixes=("", "_dim"),
    )

    # Prefer warehouse interest name if present; fall back to fact column
    if "interest_name" in merged.columns:
        merged["area_of_interest"] = merged["interest_name"].fillna(
            merged["area_of_interest"]
        )

    clean = pd.DataFrame(
        {
            "student_id": merged["student_id"],
            "cgpa": merged["cgpa"].astype(float),
            "ug_branch": merged["ug_branch"].astype(str).str.strip(),
            "projects_count": merged["projects_count"].astype(int),
            "internships_count": merged["internships_count"].astype(int),
            "skills": merged["skills"].astype(str).str.strip(),
            "area_of_interest": merged["area_of_interest"].astype(str).str.strip(),
            "career_goal": merged["career_goal"].astype(str).str.strip(),
            "chosen_masters_program": merged["program_name"].astype(str).str.strip(),
            "program_id": merged["program_id"],
            "field": merged["field"],
            "duration_years": merged["duration_years"],
            "example_universities": merged["example_universities"],
            "region": merged["region"],
            "category": merged["category"],
        }
    )

    # Basic quality checks
    clean = clean.dropna(subset=["cgpa", "chosen_masters_program"])
    clean = clean[(clean["cgpa"] >= 0) & (clean["cgpa"] <= 10)]
    clean = clean.drop_duplicates(subset=["student_id"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_PATH, index=False)
    print(f"ETL complete: {len(clean)} rows -> {OUTPUT_PATH}")
    return clean


if __name__ == "__main__":
    run_etl()
